#include "lower.h"

#include <vector>
#include <stack>
#include <set>

#include "taco/tensor.h"
#include "taco/expr.h"

#include "ir/ir.h"
#include "ir/ir_visitor.h"
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

static Iterator getIterator(std::vector<storage::Iterator>& iterators) {
  taco_iassert(!iterators.empty());

  Iterator iter = iterators[0];
  for (size_t i = 1; i < iterators.size(); ++i) {
    if (!iterators[i].isRandomAccess()) {
      iter = iterators[i];
    }
  }
  return iter;
}

IndexExpr emitAvailableExprs(const IndexVar& indexVar, const IndexExpr& indexExpr,
                             Context* ctx, vector<Stmt>* stmts) {
  vector<IndexVar>  visited    = ctx->schedule.getAncestors(indexVar);
  vector<IndexExpr> availExprs = getAvailableExpressions(indexExpr, visited);
  map<IndexExpr,IndexExpr> substitutions;
  for (const IndexExpr& availExpr : availExprs) {
    TensorBase t("t" + indexVar.getName(), ComponentType::Double);
    substitutions.insert({availExpr, taco::Access(t)});
    Expr tensorVar = Var::make(t.getName(), Type(Type::Float,64));
    ctx->temporaries.insert({t, tensorVar});
    Expr expr = lowerToScalarExpression(availExpr, ctx->iterators,
                                        ctx->schedule, ctx->temporaries);
    stmts->push_back(VarAssign::make(tensorVar, expr, true));
  }
  return replace(indexExpr, substitutions);
}

void emitComputeExpr(const Target& target,
                     const IndexVar& indexVar, const IndexExpr& indexExpr,
                     const Context& ctx, vector<Stmt>* stmts) {
  Expr expr = lowerToScalarExpression(indexExpr, ctx.iterators,
                                      ctx.schedule, ctx.temporaries);
  if (target.pos.defined()) {
    Stmt store = ctx.schedule.hasReductionVariableAncestor(indexVar)
        ? compoundStore(target.tensor, target.pos, expr)
        :   Store::make(target.tensor, target.pos, expr);
    stmts->push_back(store);
  }
  else {
    Stmt assign = ctx.schedule.hasReductionVariableAncestor(indexVar)
        ?  compoundAssign(target.tensor, expr)
        : VarAssign::make(target.tensor, expr);
    stmts->push_back(assign);
  }
}

bool isParallelizable(const IndexVar& indexVar, const Context& ctx) {
  TensorPath resultPath = ctx.schedule.getResultTensorPath();
  for (size_t i = 0; i < resultPath.getSize(); i++){
    if (!ctx.iterators[resultPath.getStep(i)].isDense()) {
      return false;
    }
  }
  return ctx.schedule.getAncestors(indexVar).size() == 1 &&
         ctx.schedule.isFree(indexVar);
}

static vector<Stmt> lower(const Target&    target,
                          const IndexExpr& indexExpr,
                          const IndexVar&  indexVar,
                          Context&         ctx) {
  vector<Stmt> code;

  MergeLattice lattice = MergeLattice::make(indexExpr, indexVar, ctx.schedule,
                                            ctx.iterators);
  auto         latticeIterators = lattice.getIterators();

  TensorPath     resultPath     = ctx.schedule.getResultTensorPath();
  TensorPathStep resultStep     = resultPath.getStep(indexVar);
  Iterator       resultIterator = (resultStep.getPath().defined())
                                  ? ctx.iterators[resultStep]
                                  : Iterator();

  bool emitCompute  = util::contains(ctx.properties, Compute);
  bool emitAssemble = util::contains(ctx.properties, Assemble);
  bool emitMerge    = needsMerge(lattice);

  // Emit code to initialize pos variables: B2_pos = B2_pos_arr[B1_pos];
  if (emitMerge) {
    for (auto& iterator : latticeIterators) {
      Expr iteratorVar = iterator.getIteratorVar();
      Stmt iteratorInit = VarAssign::make(iteratorVar, iterator.begin(), true);
      code.push_back(iteratorInit);
    }
  }

  // Emit one loop per lattice point lp
  vector<Stmt> loops;
  for (MergeLatticePoint lp : lattice) {
    MergeLattice lpLattice = lattice.getSubLattice(lp);
    auto lpIterators      = lp.getIterators();

    vector<Stmt> loopBody;

    // Emit code to initialize sequential access idx variables:
    // int kB = B2_idx_arr[B2_pos];
    vector<Expr> mergeIdxVariables;
    auto sequentialAccessIterators = getSequentialAccessIterators(lpIterators);
    for (Iterator& iterator : sequentialAccessIterators) {
      Stmt initIdx = iterator.initDerivedVar();
      loopBody.push_back(initIdx);
      mergeIdxVariables.push_back(iterator.getIdxVar());
    }

    // Emit code to initialize the index variable: k = min(kB, kc);
    Expr idx = (lp.getMergeIterators().size() > 1)
               ? min(indexVar.getName(), lp.getMergeIterators(), &loopBody)
               : lp.getMergeIterators()[0].getIdxVar();

    // Emit code to initialize random access pos variables:
    // B2_pos = (B1_pos*3) + k;
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

      // Case expression
      vector<Expr> stepIdxEqIdx;
      // TODO: Fix below iteration
      for (auto& iter : lq.getRangeIterators()) {
        stepIdxEqIdx.push_back(Eq::make(iter.getIdxVar(), idx));
      }
      Expr caseExpr = conjunction(stepIdxEqIdx);

      // Case body
      vector<Stmt> caseBody;

      // Emit compute code. Cases: ABOVE_LAST_FREE, LAST_FREE or BELOW_LAST_FREE
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
          // Reduce child expression into temporary
          TensorBase t("t" + child.getName(), ComponentType::Double);
          Expr tensorVar = Var::make(t.getName(), Type(Type::Float,64));
          ctx.temporaries.insert({t, tensorVar});
          childTarget.tensor = tensorVar;
          childTarget.pos    = Expr();
          if (emitCompute) {
            caseBody.push_back(VarAssign::make(tensorVar, 0.0, true));
          }

          // Extract the expression to compute at the next level
          childExpr = getSubExpr(lqExpr, ctx.schedule.getDescendants(child));

          // Rewrite lqExpr to substitute the expression computed at the next
          // level with the temporary
          lqExpr = replace(lqExpr, {{childExpr,taco::Access(t)}});
        }
        taco_iassert(childExpr.defined());
        auto childCode = lower::lower(childTarget, childExpr, child, ctx);
        util::append(caseBody, childCode);
      }

      // Emit code to compute and store/assign result 
      if (emitCompute &&
          (indexVarCase == LAST_FREE || indexVarCase == BELOW_LAST_FREE)) {
        emitComputeExpr(target, indexVar, lqExpr, ctx, &caseBody);
      }

      // Emit a store of the index variable value to the result idx index array
      // A2_idx_arr[A2_pos] = j;
      if (emitAssemble && resultIterator.defined()){
        Stmt idxStore = resultIterator.storeIdx(idx);
        if (idxStore.defined()) {
          caseBody.push_back(idxStore);
        }
      }

      // Emit code to increment the results iterator variable
      if (resultIterator.defined() && resultIterator.isSequentialAccess()) {
        Expr resultPos = resultIterator.getPtrVar();
        Stmt posInc = VarAssign::make(resultPos, Add::make(resultPos, 1));

        Expr doResize = ir::And::make(
            Eq::make(0, BitAnd::make(Add::make(resultPos, 1), resultPos)),
            Lte::make(ctx.allocSize, Add::make(resultPos, 1)));
        Expr newSize = ir::Mul::make(2, ir::Add::make(resultPos, 1));
        Stmt resizeIndices = resultIterator.resizeIdxStorage(newSize);

        if (resultStep != resultPath.getLastStep()) {
          // Emit code to resize idx and pos
          if (emitAssemble) {
            auto nextStep = resultPath.getStep(resultStep.getStep()+1);
            Iterator iterNext = ctx.iterators[nextStep];
            Stmt resizePos = iterNext.resizePtrStorage(newSize);
            if (resizePos.defined()) {
              resizeIndices = Block::make({resizeIndices, resizePos});
            }
            resizeIndices = IfThenElse::make(doResize, resizeIndices);
            posInc = Block::make({posInc, resizeIndices});
          }

          int step = resultStep.getStep() + 1;
          string resultTensorName = resultIterator.getTensor().as<Var>()->name;
          string posArrName = resultTensorName + util::toString(step) +
                              "_pos_arr";
          Expr posArr = GetProperty::make(resultIterator.getTensor(),
                                          TensorProperty::Indices,
                                          step, 0, posArrName);

          Expr producedVals =
              Gt::make(Load::make(posArr, Add::make(resultPos,1)),
                       Load::make(posArr, resultPos));
          posInc = IfThenElse::make(producedVals, posInc);
        } else if (emitAssemble) {
          // Emit code to resize idx (at result store loop nest)
          resizeIndices = IfThenElse::make(doResize, resizeIndices);
          posInc = Block::make({posInc, resizeIndices});
        }
        util::append(caseBody, {posInc});
      }
      cases.push_back({caseExpr, Block::make(caseBody)});
    }
    loopBody.push_back(needsMerge(lpLattice)
                       ? Case::make(cases, lpLattice.isFull())
                       : cases[0].second);

    // Emit code to conditionally increment sequential access pos variables
    if (emitMerge) {
      vector<Stmt> incs;
      vector<Stmt> maybeIncs;
      for (Iterator& iterator : lpIterators) {
        Expr pos = iterator.getIteratorVar();
        Stmt inc = VarAssign::make(pos, Add::make(pos, 1));
        Expr tensorIdx = iterator.getIdxVar();
        if (!iterator.isDense() && iterator.getIdxVar() != idx) {
          maybeIncs.push_back(IfThenElse::make(Eq::make(tensorIdx, idx), inc));
        }
        else {
          incs.push_back(inc);
        }
      }
      util::append(loopBody, maybeIncs);
      util::append(loopBody, incs);
    }

    // Emit loop (while loop for merges and for loop for non-merges)
    Stmt loop;
    if (emitMerge) {
      // Loop until any index has been exchaused
      vector<Expr> stepIterLqEnd;
      for (auto& iter : lp.getRangeIterators()) {
        stepIterLqEnd.push_back(Lt::make(iter.getIteratorVar(), iter.end()));
      }
      Expr untilAnyExhausted = conjunction(stepIterLqEnd);
      loop = While::make(untilAnyExhausted, Block::make(loopBody));
    }
    else {
      LoopKind loopKind = isParallelizable(indexVar, ctx) ? LoopKind::Parallel
                                                          : LoopKind::Serial;
      taco_iassert(lp.getMergeIterators().size() == 1);
      Iterator iter = lp.getMergeIterators()[0];
      loop = For::make(iter.getIteratorVar(), iter.begin(), iter.end(), 1,
                       Block::make(loopBody), loopKind);
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
  auto vars = tensor.getIndexVars();
  auto indexExpr = tensor.getExpr();

  // Pack the tensor and it's expression operands into the parameter list
  vector<Expr> parameters;
  vector<Expr> results;
  map<TensorBase,Expr> tensorVars;
  tie(parameters,results,tensorVars) = getTensorVars(tensor);

  // Create the schedule and the iterators of the lowered code
  ctx.schedule = IterationSchedule::make(tensor);
  ctx.iterators = Iterators(ctx.schedule, tensorVars);

  vector<Stmt> body;

  TensorPath resultPath = ctx.schedule.getResultTensorPath();
  if (emitAssemble) {
    for (auto& indexVar : tensor.getIndexVars()) {
      Iterator iter = ctx.iterators[resultPath.getStep(indexVar)];
      Stmt allocStmts = iter.initStorage(ctx.allocSize);
      if (allocStmts.defined()) {
        if (body.empty()) {
          const auto comment = to<Var>(ctx.allocSize)->name + 
                               " should be initialized to a power of two";
          Stmt setAllocSize = Block::make({
              Comment::make(comment),
              VarAssign::make(ctx.allocSize, tensor.getAllocSize(), true)
          });
          body.push_back(setAllocSize);
        }
        body.push_back(allocStmts);
      }
    }

    if (!body.empty()) {
      body.push_back(BlankLine::make());
    }
  }

  // Initialize the result pos variables
  Stmt prevIteratorInit;
  for (auto& indexVar : tensor.getIndexVars()) {
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

    const bool emitLoops = emitCompute || [&]() {
      for (auto& indexVar : tensor.getIndexVars()) {
        Iterator iter = ctx.iterators[resultPath.getStep(indexVar)];
        if (!iter.isDense()) {
          return true;
        }
      }
      return false;
    }();
    if (emitLoops) {
      for (auto& root : roots) {
        auto loopNest = lower::lower(target, indexExpr, root, ctx);
        util::append(body, loopNest);
      }
    }

    if (emitAssemble) {
      Expr size = 1;
      for (auto& indexVar : tensor.getIndexVars()) {
        Iterator iter = ctx.iterators[resultPath.getStep(indexVar)];
        size = iter.isFixedRange() ? Mul::make(size, iter.end()) : 
               iter.getPtrVar();
      }
      Stmt allocVals = Allocate::make(target.tensor, size);
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
      body.push_back(allocVals);
    }
    if (emitCompute) {
      Expr expr = lowerToScalarExpression(indexExpr, ctx.iterators, ctx.schedule,
                                          map<TensorBase,Expr>());
      Stmt compute = Store::make(vals, 0, expr);
      body.push_back(compute);
    }
  }

  return Function::make(funcName, parameters, results, Block::make(body));
}
}}
