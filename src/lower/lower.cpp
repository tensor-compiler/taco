#include "taco/lower/lower.h"

#include <vector>
#include <stack>
#include <set>

#include "taco/tensor.h"
#include "taco/expr.h"

#include "taco/ir/ir.h"
#include "taco/ir/ir_visitor.h"
#include "ir/ir_codegen.h"

#include "lower_codegen.h"
#include "iterators.h"
#include "tensor_path.h"
#include "merge_lattice.h"
#include "iteration_schedule.h"
#include "expr_tools.h"
#include "taco/expr_nodes/expr_nodes.h"
#include "taco/expr_nodes/expr_rewriter.h"
#include "storage/iterator.h"
#include "taco/util/name_generator.h"
#include "taco/util/collections.h"
#include "taco/util/strings.h"

using namespace std;

namespace taco {
namespace lower {

using namespace taco::ir;
using namespace taco::expr_nodes;
using taco::storage::Iterator;

struct Context {
  /// Determines what kind of code to emit (e.g. compute and/or assembly)
  set<Property>        properties;

  /// The iteration schedule to use for lowering the index expression
  IterationSchedule    schedule;

  /// The iterators of the tensor tree levels
  Iterators            iterators;

  /// The size of initial memory allocations
  Expr                 allocSize;

  /// Maps tensor (scalar) temporaries to IR variables.
  /// (Not clear if this approach to temporaries is too hacky.)
  map<TensorBase,Expr> temporaries;
};

struct Target {
  Expr tensor;
  Expr pos;
};

enum ComputeCase {
  // Emit the last free variable. We first recurse to compute remaining
  // reduction variables into a temporary, before we compute and store the
  // main expression
  LAST_FREE,

  // Emit a variable above the last free variable. First emit code to compute
  // available expressions and store them in temporaries, before
  // we recurse on the next index variable.
  ABOVE_LAST_FREE,

  // Emit a variable below the last free variable. First recurse to emit
  // remaining (summation) variables, before we add in the available expressions
  // for the current summation variable.
  BELOW_LAST_FREE
};

static ComputeCase getComputeCase(const IndexVar& indexVar,
                                  const IterationSchedule& schedule) {
  if (schedule.isLastFreeVariable(indexVar)) {
    return LAST_FREE;
  }
  else if (schedule.hasFreeVariableDescendant(indexVar)) {
    return ABOVE_LAST_FREE;
  }
  else {
    return BELOW_LAST_FREE;
  }
}

/// Returns true iff the lattice must be merged, false otherwise. A lattice
/// must be merged iff it has more than one lattice point, or two or more of
/// its iterators are not random access.
static bool needsMerge(MergeLattice lattice) {
  if (lattice.getSize() > 1) {
    return true;
  }

  int notRandomAccess = 0;
  for (auto& iterator : lattice.getIterators()) {
    if ((!iterator.isRandomAccess()) && (++notRandomAccess > 1)) {
      return true;
    }
  }
  return false;
}

static bool needsZero(const Context& ctx) {
  const auto& schedule = ctx.schedule;
  const auto& resultIdxVars = schedule.getResultTensorPath().getVariables();

  if (schedule.hasReductionVariableAncestor(resultIdxVars.back())) {
    return true;
  }

  for (const auto& idxVar : resultIdxVars) {
    for (const auto& tensorPath : schedule.getTensorPaths()) {
      if (util::contains(tensorPath.getVariables(), idxVar) && 
          !ctx.iterators[tensorPath.getStep(idxVar)].isDense()) {
        return true;
      }
    }
  }
  return false;
}

static IndexExpr emitAvailableExprs(const IndexVar& indexVar,
                                    const IndexExpr& indexExpr, Context* ctx,
                                    vector<Stmt>* stmts) {
  vector<IndexVar>  visited    = ctx->schedule.getAncestors(indexVar);
  vector<IndexExpr> availExprs = getAvailableExpressions(indexExpr, visited);
  map<IndexExpr,IndexExpr> substitutions;
  for (const IndexExpr& availExpr : availExprs) {
    TensorBase t("t" + indexVar.getName(), Float(64));
    substitutions.insert({availExpr, taco::Access(t)});
    Expr tensorVar = Var::make(t.getName(), Float(64));
    ctx->temporaries.insert({t, tensorVar});
    Expr expr = lowerToScalarExpression(availExpr, ctx->iterators,
                                        ctx->schedule, ctx->temporaries);
    stmts->push_back(VarAssign::make(tensorVar, expr, true));
  }
  return replace(indexExpr, substitutions);
}

static void emitComputeExpr(const Target& target, const IndexVar& indexVar,
                            const IndexExpr& indexExpr, const Context& ctx,
                            vector<Stmt>* stmts, bool accum) {
  Expr expr = lowerToScalarExpression(indexExpr, ctx.iterators,
                                      ctx.schedule, ctx.temporaries);
  if (target.pos.defined()) {
    Stmt store = ctx.schedule.hasReductionVariableAncestor(indexVar) || accum
        ? compoundStore(target.tensor, target.pos, expr)
        :   Store::make(target.tensor, target.pos, expr);
    stmts->push_back(store);
  }
  else {
    Stmt assign = ctx.schedule.hasReductionVariableAncestor(indexVar) || accum
        ?  compoundAssign(target.tensor, expr)
        : VarAssign::make(target.tensor, expr);
    stmts->push_back(assign);
  }
}

static LoopKind doParallelize(const IndexVar& indexVar, const Expr& tensor, 
                              const Context& ctx) {
  if (ctx.schedule.getAncestors(indexVar).size() != 1 ||
      ctx.schedule.isReduction(indexVar)) {
    return LoopKind::Serial;
  }

  const TensorPath& resultPath = ctx.schedule.getResultTensorPath();
  for (size_t i = 0; i < resultPath.getSize(); i++){
    if (!ctx.iterators[resultPath.getStep(i)].isDense()) {
      return LoopKind::Serial;
    }
  }

  const TensorPath parallelizedAccess = [&]() {
    const auto tensorName = tensor.as<Var>()->name;
    for (const auto& tensorPath : ctx.schedule.getTensorPaths()) {
      if (tensorPath.getTensor().getName() == tensorName) {
        return tensorPath;
      }
    }
    taco_iassert(false);
    return TensorPath();
  }();

  if (parallelizedAccess.getSize() <= 2) {
    return LoopKind::Static;
  }

  for (size_t i = 1; i < parallelizedAccess.getSize(); ++i) {
    if (ctx.iterators[parallelizedAccess.getStep(i)].isDense()) {
      return LoopKind::Static;
    }
  }

  return LoopKind::Dynamic;
}

/// Expression evaluates to true iff none of the iteratators are exhausted
static Expr noneExhausted(const vector<Iterator>& iterators) {
  vector<Expr> stepIterLqEnd;
  for (auto& iter : iterators) {
    stepIterLqEnd.push_back(Lt::make(iter.getIteratorVar(), iter.end()));
  }
  return conjunction(stepIterLqEnd);
}

/// Expression evaluates to true iff all the iterator idx vars are equal to idx
/// or if there are no iterators.
static Expr allEqualTo(const vector<Iterator>& iterators, Expr idx) {
  if (iterators.size() == 0) {
    return Literal::make(true);
  }

  vector<Expr> iterIdxEqualToIdx;
  for (auto& iter : iterators) {
    iterIdxEqualToIdx.push_back(Eq::make(iter.getIdxVar(), idx));
  }
  return conjunction(iterIdxEqualToIdx);
}

/// Returns the iterator for the `idx` variable from `iterators`, or Iterator()
/// none of the iterator iterate over `idx`.
static Iterator getIterator(const Expr& idx,
                            const vector<Iterator>& iterators) {
  for (auto& iterator : iterators) {
    if (iterator.getIdxVar() == idx) {
      return iterator;
    }
  }
  return Iterator();
}

static vector<Iterator> removeIterator(const Expr& idx,
                                       const vector<Iterator>& iterators) {
  vector<Iterator> result;
  for (auto& iterator : iterators) {
    if (iterator.getIdxVar() != idx) {
      result.push_back(iterator);
    }
  }
  return result;
}

static Stmt createIfStatements(vector<pair<Expr,Stmt>> cases,
                               const MergeLattice& lattice) {
  if (!needsMerge(lattice)) {
    return cases[0].second;
  }

  vector<pair<Expr,Stmt>> ifCases;
  pair<Expr,Stmt> elseCase;
  for (auto& cas : cases) {
    auto lit = cas.first.as<Literal>();
    if (lit != nullptr && lit->type == Bool() && lit->value == 1){
      taco_iassert(!elseCase.first.defined()) <<
          "there should only be one true case";
      elseCase = cas;
    }
    else {
      ifCases.push_back(cas);
    }
  }

  if (elseCase.first.defined()) {
    ifCases.push_back(elseCase);
    return Case::make(ifCases, true);
  }
  else {
    return Case::make(ifCases, lattice.isFull());
  }
}

static vector<Stmt> lower(const Target&    target,
                          const IndexExpr& indexExpr,
                          const IndexVar&  indexVar,
                          Context&         ctx) {
  vector<Stmt> code;

  MergeLattice lattice = MergeLattice::make(indexExpr, indexVar, ctx.schedule,
                                            ctx.iterators);

  TensorPath     resultPath     = ctx.schedule.getResultTensorPath();
  TensorPathStep resultStep     = resultPath.getStep(indexVar);
  Iterator       resultIterator = (resultStep.getPath().defined())
                                  ? ctx.iterators[resultStep]
                                  : Iterator();

  bool accumulate   = util::contains(ctx.properties, Accumulate);
  bool emitCompute  = util::contains(ctx.properties, Compute);
  bool emitAssemble = util::contains(ctx.properties, Assemble);
  bool emitMerge    = needsMerge(lattice);

  // Emit code to initialize pos variables:
  // B2_pos = B2_pos_arr[B1_pos];
  if (emitMerge) {
    for (auto& iterator : lattice.getIterators()) {
      Expr iteratorVar = iterator.getIteratorVar();
      Stmt iteratorInit = VarAssign::make(iteratorVar, iterator.begin(), true);
      code.push_back(iteratorInit);
    }
  }

  // Emit one loop per lattice point lp
  vector<Stmt> loops;
  for (MergeLatticePoint lp : lattice) {
    MergeLattice lpLattice = lattice.getSubLattice(lp);
    auto lpIterators = lp.getIterators();

    vector<Stmt> loopBody;

    // Emit code to initialize sequential access idx variables:
    // int kB = B1_idx_arr[B1_pos];
    // int kc = c0_idx_arr[c0_pos];
    vector<Expr> mergeIdxVariables;
    auto sequentialAccessIterators = getSequentialAccessIterators(lpIterators);
    for (Iterator& iterator : sequentialAccessIterators) {
      Stmt initIdx = iterator.initDerivedVar();
      loopBody.push_back(initIdx);
      mergeIdxVariables.push_back(iterator.getIdxVar());
    }

    // Emit code to initialize the index variable:
    // k = min(kB, kc);
    Expr idx = (lp.getMergeIterators().size() > 1)
               ? min(indexVar.getName(), lp.getMergeIterators(), &loopBody)
               : lp.getMergeIterators()[0].getIdxVar();

    // Emit code to initialize random access pos variables:
    // D1_pos = (D0_pos * 3) + k;
    auto randomAccessIterators =
        getRandomAccessIterators(util::combine(lpIterators, {resultIterator}));
    for (Iterator& iterator : randomAccessIterators) {
      Expr val = ir::Add::make(ir::Mul::make(iterator.getParent().getPtrVar(),
                                             iterator.end()), idx);
      Stmt initPos = VarAssign::make(iterator.getPtrVar(), val, true);
      loopBody.push_back(initPos);
    }

    // Emit one case per lattice point in the sub-lattice rooted at lp
    vector<pair<Expr,Stmt>> cases;
    for (MergeLatticePoint& lq : lpLattice) {
      IndexExpr lqExpr = lq.getExpr();
      vector<Stmt> caseBody;

      // Emit compute code for three cases: above, at or below the last free var
      ComputeCase indexVarCase = getComputeCase(indexVar, ctx.schedule);

      // Emit available sub-expressions at this loop level
      if (emitCompute && ABOVE_LAST_FREE == indexVarCase) {
        lqExpr = emitAvailableExprs(indexVar, lqExpr, &ctx, &caseBody);
      }

      // Recursive call to emit iteration schedule children
      for (auto& child : ctx.schedule.getChildren(indexVar)) {
        IndexExpr childExpr = lqExpr;
        Target childTarget = target;
        if (indexVarCase == LAST_FREE || indexVarCase == BELOW_LAST_FREE) {
          // Extract the expression to compute at the next level. If there's no
          // computation on the next level for this lattice case then skip it
          childExpr = getSubExpr(lqExpr, ctx.schedule.getDescendants(child));
          if (!childExpr.defined()) continue;

          // Reduce child expression into temporary
          TensorBase t("t" + child.getName(), Float(64));
          Expr tensorVar = Var::make(t.getName(), Type(Type::Float,64));
          ctx.temporaries.insert({t, tensorVar});
          childTarget.tensor = tensorVar;
          childTarget.pos    = Expr();
          if (emitCompute) {
            caseBody.push_back(VarAssign::make(tensorVar, 0.0, true));
          }

          // Rewrite lqExpr to substitute the expression computed at the next
          // level with the temporary
          lqExpr = replace(lqExpr, {{childExpr,taco::Access(t)}});
        }
        auto childCode = lower::lower(childTarget, childExpr, child, ctx);
        util::append(caseBody, childCode);
      }

      // Emit code to compute and store/assign result 
      if (emitCompute &&
          (indexVarCase == LAST_FREE || indexVarCase == BELOW_LAST_FREE)) {
        emitComputeExpr(target, indexVar, lqExpr, ctx, &caseBody, accumulate);
      }

      // Emit a store of the index variable value to the result idx index array
      // A2_idx_arr[A2_pos] = j;
      if (emitAssemble && resultIterator.defined()){
        Stmt idxStore = resultIterator.storeIdx(idx);
        if (idxStore.defined()) {
          caseBody.push_back(idxStore);
        }
      }

      // Emit code to increment the result `pos` variable and to allocate
      // additional storage for result `idx` and `pos` arrays
      if (resultIterator.defined() && resultIterator.isSequentialAccess()) {
        Expr rpos = resultIterator.getPtrVar();
        Stmt posInc = VarAssign::make(rpos, Add::make(rpos, 1));

        // Conditionally resize result `idx` and `pos` arrays
        if (emitAssemble) {
          Expr resize =
              And::make(Eq::make(0, BitAnd::make(Add::make(rpos, 1), rpos)),
                        Lte::make(ctx.allocSize, Add::make(rpos, 1)));
          Expr newSize = ir::Mul::make(2, ir::Add::make(rpos, 1));

          // Resize result `idx` array
          Stmt resizeIndices = resultIterator.resizeIdxStorage(newSize);

          // Resize result `pos` array
          if (indexVarCase == ABOVE_LAST_FREE) {
            auto nextStep = resultPath.getStep(resultStep.getStep() + 1);
            Stmt resizePos = ctx.iterators[nextStep].resizePtrStorage(newSize);
            resizeIndices = Block::make({resizeIndices, resizePos});
          } else if (resultStep == resultPath.getLastStep() && emitCompute) {
            Expr vals = GetProperty::make(resultIterator.getTensor(),
                                          TensorProperty::Values);
            Stmt resizeVals = Allocate::make(vals, newSize, true);
            resizeIndices = Block::make({resizeIndices, resizeVals});
          }
          posInc = Block::make({posInc,IfThenElse::make(resize,resizeIndices)});
        }

        // Only increment `pos` if values were produced at the next level
        if (indexVarCase == ABOVE_LAST_FREE) {
          int step = resultStep.getStep() + 1;
          string resultTensorName = resultIterator.getTensor().as<Var>()->name;
          string posArrName = resultTensorName + to_string(step + 1) + "_pos";
          Expr posArr = GetProperty::make(resultIterator.getTensor(),
                                          TensorProperty::Indices,
                                          step, 0, posArrName);
          Expr producedVals = Gt::make(Load::make(posArr, Add::make(rpos,1)),
                                       Load::make(posArr, rpos));
          posInc = IfThenElse::make(producedVals, posInc);
        }
        util::append(caseBody, {posInc});
      }

      auto caseIterators = removeIterator(idx, lq.getRangeIterators());
      cases.push_back({allEqualTo(caseIterators,idx), Block::make(caseBody)});
    }
    loopBody.push_back(createIfStatements(cases, lpLattice));

    // Emit code to increment sequential access `pos` variables. Variables that
    // may not be consumed in an iteration (i.e. their iteration space is
    // different from the loop iteration space) are guarded by a conditional:
    if (emitMerge) {
      // if (k == kB) B1_pos++;
      // if (k == kc) c0_pos++;
      for (auto& iterator : removeIterator(idx, lp.getRangeIterators())) {
        Expr ivar = iterator.getIteratorVar();
        Stmt inc = VarAssign::make(ivar, Add::make(ivar, 1));
        Expr tensorIdx = iterator.getIdxVar();
        loopBody.push_back(IfThenElse::make(Eq::make(tensorIdx, idx), inc));
      }

      /// k++
      auto idxIterator = getIterator(idx, lpIterators);
      if (idxIterator.defined()) {
        Expr ivar = idxIterator.getIteratorVar();
        loopBody.push_back(VarAssign::make(ivar, Add::make(ivar, 1)));
      }
    }

    // Emit loop (while loop for merges and for loop for non-merges)
    Stmt loop;
    if (emitMerge) {
      loop = While::make(noneExhausted(lp.getRangeIterators()),
                         Block::make(loopBody));
    }
    else {
      Iterator iter = lp.getRangeIterators()[0];
      loop = For::make(iter.getIteratorVar(), iter.begin(), iter.end(), 1,
                       Block::make(loopBody), 
                       doParallelize(indexVar, iter.getTensor(), ctx));
    }
    loops.push_back(loop);
  }
  util::append(code, loops);

  // Emit a store of the  segment size to the result pos index
  // A2_pos_arr[A1_pos + 1] = A2_pos;
  if (emitAssemble && resultIterator.defined()) {
    Stmt posStore = resultIterator.storePtr();
    if (posStore.defined()) {
      util::append(code, {posStore});
    }
  }

  return code;
}

Stmt lower(TensorBase tensor, string funcName, set<Property> properties) {
  Context ctx;
  ctx.allocSize  = Var::make("init_alloc_size", Type(Type::Int));
  ctx.properties = properties;

  const bool emitAssemble = util::contains(ctx.properties, Assemble);
  const bool emitCompute = util::contains(ctx.properties, Compute);

  auto name = tensor.getName();
  auto indexExpr = tensor.getExpr();

  // Pack the tensor and it's expression operands into the parameter list
  vector<Expr> parameters;
  vector<Expr> results;
  map<TensorBase,Expr> tensorVars;
  tie(parameters,results,tensorVars) = getTensorVars(tensor);

  // Create the schedule and the iterators of the lowered code
  ctx.schedule = IterationSchedule::make(tensor);
  ctx.iterators = Iterators(ctx.schedule, tensorVars);

  vector<Stmt> init, body;

  TensorPath resultPath = ctx.schedule.getResultTensorPath();
  if (emitAssemble) {
    for (auto& indexVar : resultPath.getVariables()) {
      Iterator iter = ctx.iterators[resultPath.getStep(indexVar)];
      Stmt allocStmts = iter.initStorage(ctx.allocSize);
      if (allocStmts.defined()) {
        if (init.empty()) {
          const auto comment = to<Var>(ctx.allocSize)->name + 
                               " should be initialized to a power of two";
          Stmt setAllocSize = Block::make({
              Comment::make(comment),
              VarAssign::make(ctx.allocSize, (int)tensor.getAllocSize(), true)
          });
          init.push_back(setAllocSize);
        }
        init.push_back(allocStmts);
      }
    }
  }

  // Initialize the result pos variables
  if (emitCompute || emitAssemble) {
    Stmt prevIteratorInit;
    for (auto& indexVar : resultPath.getVariables()) {
      Iterator iter = ctx.iterators[resultPath.getStep(indexVar)];
      Stmt iteratorInit = VarAssign::make(iter.getPtrVar(), iter.begin(), true);
      if (iter.isSequentialAccess()) {
        // Emit code to initialize the result pos variable
        if (prevIteratorInit.defined()) {
          body.push_back(prevIteratorInit);
          prevIteratorInit = Stmt();
        }
        body.push_back(iteratorInit);
      } else {
        prevIteratorInit = iteratorInit;
      }
    }
  }
  taco_iassert(results.size() == 1) << "An expression can only have one result";

  // Lower the iteration schedule
  auto& roots = ctx.schedule.getRoots();

  // Lower tensor expressions
  if (roots.size() > 0) {
    Iterator resultIterator = (resultPath.getSize() > 0)
        ? ctx.iterators[resultPath.getLastStep()]
        : ctx.iterators.getRoot(resultPath);  // e.g. `a = b(i) * c(i)`
    Target target;
    target.tensor = GetProperty::make(resultIterator.getTensor(),
                                      TensorProperty::Values);
    target.pos = resultIterator.getPtrVar();

    if (emitCompute) {
      Expr size = 1;
      for (auto& indexVar : resultPath.getVariables()) {
        const Iterator iter = ctx.iterators[resultPath.getStep(indexVar)];
        if (!iter.isFixedRange()) {
          size = ctx.allocSize;
          break;
        }
        size = Mul::make(size, iter.end());
      }

      if (emitAssemble) {
        Stmt allocVals = Allocate::make(target.tensor, size);
        init.push_back(allocVals);
      }

      // Emit code to zero result value array, if the output is dense and if
      // either an output mode is merged with a sparse input mode or if the
      // emitted code is a scatter code.
      if (!isa<Var>(size) && !util::contains(properties, Accumulate)) {
        if (isa<Literal>(size)) {
          taco_iassert(to<Literal>(size)->value == 1);
          body.push_back(Store::make(target.tensor, 0, 0.0));
        } else if (needsZero(ctx)) {
          Expr idxVar = Var::make("p" + name, Type(Type::Int));
          Stmt zeroStmt = Store::make(target.tensor, idxVar, 0.0);
          body.push_back(For::make(idxVar, 0, size, 1, zeroStmt));
        }
      }
    }

    const bool emitLoops = emitCompute || (emitAssemble && [&]() {
      for (auto& indexVar : resultPath.getVariables()) {
        Iterator iter = ctx.iterators[resultPath.getStep(indexVar)];
        if (!iter.isDense()) {
          return true;
        }
      }
      return false;
    }());
    if (emitLoops) {
      for (auto& root : roots) {
        auto loopNest = lower::lower(target, indexExpr, root, ctx);
        util::append(body, loopNest);
      }
    }

    if (emitAssemble && !emitCompute) {
      Expr size = 1;
      for (auto& indexVar : resultPath.getVariables()) {
        Iterator iter = ctx.iterators[resultPath.getStep(indexVar)];
        size = iter.isFixedRange() ? Mul::make(size, iter.end()) : 
               iter.getPtrVar();
      }
      Stmt allocVals = Allocate::make(target.tensor, size);
      
      if (!body.empty()) {
        body.push_back(BlankLine::make());
      }
      body.push_back(allocVals);
    }
  }
  // Lower scalar expressions
  else {
    TensorPath resultPath = ctx.schedule.getResultTensorPath();
    Expr resultTensorVar = ctx.iterators.getRoot(resultPath).getTensor();
    Expr vals = GetProperty::make(resultTensorVar, TensorProperty::Values);
    if (emitAssemble) {
      Stmt allocVals = Allocate::make(vals, 1);
      init.push_back(allocVals);
    }
    if (emitCompute) {
      Expr expr = lowerToScalarExpression(indexExpr, ctx.iterators, ctx.schedule,
                                          map<TensorBase,Expr>());
      Stmt compute = Store::make(vals, 0, expr);
      body.push_back(compute);
    }
  }

  if (!init.empty()) {
    init.push_back(BlankLine::make());
  }
  body = util::combine(init, body);

  return Function::make(funcName, parameters, results, Block::make(body));
}
}}
