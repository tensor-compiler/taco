#include "lower.h"

#include <vector>
#include <stack>
#include <set>

#include "taco/tensor_base.h"
#include "taco/expr.h"
#include "taco/operator.h"

#include "ir/ir.h"
#include "ir/ir_visitor.h"
#include "ir/ir_codegen.h"

#include "lower_codegen.h"
#include "iterators.h"
#include "tensor_path.h"
#include "merge_lattice.h"
#include "iteration_schedule.h"
#include "available_exprs.h"
#include "taco/expr_nodes/expr_rewriter.h"
#include "storage/iterator.h"
#include "taco/util/name_generator.h"
#include "taco/util/collections.h"
#include "taco/util/strings.h"

using namespace std;

namespace taco {
namespace lower {

using namespace taco::ir;

using taco::ir::Expr;
using taco::ir::Var;
using taco::ir::Add;
using taco::storage::Iterator;

struct Context {
  /// Determines what kind of code to emit (e.g. compute and/or assembly)
  set<Property>        properties;

  /// The iteration schedule to use for lowering the index expression
  IterationSchedule    schedule;

  /// The iterators of the tensor tree levels
  Iterators            iterators;

  /// The size of initial memory allocations
  size_t               allocSize;

  /// Maps tensor (scalar) temporaries to IR variables.
  /// (Not clear if this approach to temporaries is too hacky.)
  map<TensorBase,Expr> temporaries;
};

struct Target {
  ir::Expr tensor;
  ir::Expr ptr;
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

static
ComputeCase getComputeCase(const taco::Var& indexVar,
                           const IterationSchedule& schedule) {
  ComputeCase computeCase;
  if (schedule.isLastFreeVariable(indexVar)) {
    computeCase = LAST_FREE;
  }
  else if (schedule.hasFreeVariableDescendant(indexVar)) {
    computeCase = ABOVE_LAST_FREE;
  }
  else {
    computeCase = BELOW_LAST_FREE;
  }
  return computeCase;
}

/// Returns true iff the lattice must be merged, false otherwise. A lattice
/// must be merged iff it has more than one lattice point, or two or more of
/// it's iterators are not random access.
static bool needsMerge(MergeLattice lattice) {
  if (lattice.getSize() > 1) {
    return true;
  }

  auto iterators = lattice.getIterators();
  int notRandomAccess = 0;
  for (auto& iterator : iterators) {
    if ((!iterator.isRandomAccess()) && (++notRandomAccess > 1)) {
      return true;
    }
  }
  return false;
}

// Retrieves the minimal sub-expression that covers all the index variables
static taco::Expr getSubExpr(taco::Expr expr, const vector<taco::Var>& vars) {
  class SubExprVisitor : public expr_nodes::ExprVisitor {
  public:
    SubExprVisitor(const vector<taco::Var>& vars) {
      this->vars.insert(vars.begin(), vars.end());
    }

    taco::Expr getSubExpression(const taco::Expr& expr) {
      visit(expr);
      taco::Expr e = subExpr;
      subExpr = taco::Expr();
      return e;
    }

  private:
    set<taco::Var> vars;
    taco::Expr     subExpr;

    using taco::expr_nodes::ExprVisitorStrict::visit;

    void visit(const expr_nodes::ReadNode* op) {
      for (auto& indexVar : op->indexVars) {
        if (util::contains(vars, indexVar)) {
          subExpr = op;
          return;
        }
      }
      subExpr = taco::Expr();
    }

    void visit(const expr_nodes::UnaryExprNode* op) {
      taco::Expr a = getSubExpression(op->a);
      if (a.defined()) {
        subExpr = a;
      }
      else {
        subExpr = taco::Expr();
      }
    }

    void visit(const expr_nodes::BinaryExprNode* op) {
      taco::Expr a = getSubExpression(op->a);
      taco::Expr b = getSubExpression(op->b);
      if (a.defined() && b.defined()) {
        subExpr = op;
      }
      else if (a.defined()) {
        subExpr = a;
      }
      else if (b.defined()) {
        subExpr = b;
      }
      else {
        subExpr = taco::Expr();
      }
    }

    void visit(const expr_nodes::ImmExprNode* op) {
      subExpr = op;
    }

  };
  return SubExprVisitor(vars).getSubExpression(expr);
}

static vector<Stmt> lower(const Target&     target,
                          const taco::Expr& indexExpr,
                          const taco::Var&  indexVar,
                          Context&          ctx) {
  vector<Stmt> code;
//  code.push_back(Comment::make(util::fill(toString(indexVar), '-', 70)));

  MergeLattice lattice = MergeLattice::make(indexExpr, indexVar, ctx.schedule,
                                            ctx.iterators);
  auto         latticeIterators = lattice.getIterators();

  TensorPath        resultPath     = ctx.schedule.getResultTensorPath();
  TensorPathStep    resultStep     = resultPath.getStep(indexVar);
  Iterator          resultIterator = (resultStep.getPath().defined())
                                     ? ctx.iterators[resultStep]
                                     : Iterator();

  bool emitCompute  = util::contains(ctx.properties, Compute);
  bool emitAssemble = util::contains(ctx.properties, Assemble);
  bool emitMerge    = needsMerge(lattice);

  // Emit code to initialize ptr variables: B2_ptr = B.d2.ptr[B1_ptr];
  if (emitMerge) {
    for (auto& iterator : latticeIterators) {
      Expr ptr = iterator.getPtrVar();
      Expr ptrPrev = iterator.getParent().getPtrVar();
      Expr tvar = iterator.getTensor();
      Expr iteratorVar = iterator.getIteratorVar();
      Stmt iteratorInit = VarAssign::make(iteratorVar, iterator.begin(), true);
      code.push_back(iteratorInit);
    }
  }

  // Emit one loop per lattice point lp
  vector<Stmt> loops;
  for (MergeLatticePoint lp : lattice) {
    vector<Stmt> loopBody;

    auto lpIterators = lp.getIterators();

    // Emit code to initialize sequential access idx variables:
    // kB = B.d2.idx[B2_ptr];
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

    // Emit code to initialize random access ptr variables:
    // B2_ptr = (B1_ptr*3) + k;
    auto randomAccessIterators =
        getRandomAccessIterators(util::combine(lpIterators, {resultIterator}));
    for (Iterator& iterator : randomAccessIterators) {
      Expr val = ir::Add::make(ir::Mul::make(iterator.getParent().getPtrVar(),
                                             iterator.end()), idx);
      Stmt initPtr = VarAssign::make(iterator.getPtrVar(), val, true);
      loopBody.push_back(initPtr);
    }
    loopBody.push_back(BlankLine::make());

    // Emit one case per lattice point in the sub-lattice rooted at lp
    MergeLattice lpLattice = lattice.getSubLattice(lp);
    vector<pair<Expr,Stmt>> cases;
    for (MergeLatticePoint& lq : lpLattice) {
      taco::Expr lqExpr = lq.getExpr();

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
      ComputeCase computeCase = getComputeCase(indexVar, ctx.schedule);

      // Emit available sub-expressions at this level
      if (ABOVE_LAST_FREE == computeCase && emitCompute) {
        vector<taco::Var>  visited    = ctx.schedule.getAncestors(indexVar);
        vector<taco::Expr> availExprs = getAvailableExpressions(lqExpr,visited);

        map<taco::Expr,taco::Expr> substitutions;
        for (const taco::Expr& availExpr : availExprs) {
          // Ignore expressions we've already emitted in a higher loop
          if (isa<Read>(availExpr) &&
              util::contains(ctx.temporaries,to<Read>(availExpr).getTensor())) {
            continue;
          }

          TensorBase t(util::uniqueName("t"), ComponentType::Double);
          substitutions.insert({availExpr, taco::Read(t)});

          Expr tensorVar = Var::make(t.getName(), typeOf<double>());
          ctx.temporaries.insert({t, tensorVar});

          Expr availIRExpr = lowerToScalarExpression(availExpr, ctx.iterators,
                                                     ctx.schedule,
                                                     ctx.temporaries);
          caseBody.push_back(VarAssign::make(tensorVar, availIRExpr, true));
        }
        lqExpr = expr_nodes::replace(lqExpr, substitutions);
      }

      // Recursive call to emit iteration schedule children
      for (auto& child : ctx.schedule.getChildren(indexVar)) {
        taco::Expr childExpr;
        Target childTarget;
        switch (computeCase) {
          case ABOVE_LAST_FREE: {
            childTarget.tensor = target.tensor;
            childTarget.ptr    = target.ptr;
            childExpr = lqExpr;
            break;
          }
          case LAST_FREE:
          case BELOW_LAST_FREE: {
            TensorBase t( "t" + child.getName(), ComponentType::Double);
            Expr tensorVar = Var::make(t.getName(), typeOf<double>());
            ctx.temporaries.insert({t, tensorVar});

            // Extract the expression to compute at the next level
            childExpr = getSubExpr(lqExpr, ctx.schedule.getDescendants(child));

            // Rewrite lqExpr to substitute the expression computed at the next
            // level with the temporary
            lqExpr = expr_nodes::replace(lqExpr, {{childExpr,taco::Read(t)}});

            // Reduce child expression into temporary
            util::append(caseBody, {VarAssign::make(tensorVar, 0.0, true)});
            childTarget.tensor = tensorVar;
            childTarget.ptr    = Expr();
            break;
          }
        }
        taco_iassert(childExpr.defined());
        auto childCode = lower(childTarget, childExpr, child, ctx);
        util::append(caseBody, childCode);
      }

      // Emit code to compute and store/assign result 
      if (emitCompute) {
        switch (computeCase) {
          case ABOVE_LAST_FREE:
            // Nothing to do
            break;
          case LAST_FREE:
          case BELOW_LAST_FREE: {
            Expr scalarExpr = lowerToScalarExpression(lqExpr, ctx.iterators,
                                                      ctx.schedule,
                                                      ctx.temporaries);
            if (target.ptr.defined()) {
              Stmt store = ctx.schedule.hasReductionVariableAncestor(indexVar)
                  ? compoundStore(target.tensor, target.ptr, scalarExpr)
                  :   Store::make(target.tensor, target.ptr, scalarExpr);
              caseBody.push_back(store);
            }
            else {
              Stmt assign = ctx.schedule.hasReductionVariableAncestor(indexVar)
                  ?  compoundAssign(target.tensor, scalarExpr)
                  : VarAssign::make(target.tensor, scalarExpr);
              caseBody.push_back(assign);
            }
            break;
          }
        }
      }

      // Emit a store of the index variable value to the result idx index array
      // A.d2.idx[A2_ptr] = j;
      if (emitAssemble && resultIterator.defined()){
        Stmt idxStore = resultIterator.storeIdx(idx);
        if (idxStore.defined()) {
          util::append(caseBody, {idxStore});
        }
      }

      // Emit code to increment the results iterator variable
      if (resultIterator.defined() && resultIterator.isSequentialAccess()) {
        Expr resultPtr = resultIterator.getPtrVar();
        Stmt ptrInc = VarAssign::make(resultPtr, Add::make(resultPtr, 1));

        Expr doResize = ir::And::make(
            Eq::make(0, BitAnd::make(Add::make(resultPtr, 1), resultPtr)),
            Lte::make(ctx.allocSize, Add::make(resultPtr, 1)));
        Expr newSize = ir::Mul::make(2, ir::Add::make(resultPtr, 1));
        Stmt resizeIndices = resultIterator.resizeIdxStorage(newSize);

        if (resultStep != resultPath.getLastStep()) {
          // Emit code to resize idx and ptr
          if (emitAssemble) {
            auto nextStep = resultPath.getStep(resultStep.getStep()+1);
            Iterator iterNext = ctx.iterators[nextStep];
            Stmt resizePtr = iterNext.resizePtrStorage(newSize);
            resizeIndices = Block::make({resizeIndices, resizePtr});
            resizeIndices = IfThenElse::make(doResize, resizeIndices);
            ptrInc = Block::make({ptrInc, resizeIndices});
          }

          Expr ptrArr = GetProperty::make(resultIterator.getTensor(),
                                          TensorProperty::Pointer,
                                          resultStep.getStep()+1);
          Expr producedVals =
              Gt::make(Load::make(ptrArr, Add::make(resultPtr,1)),
                       Load::make(ptrArr, resultPtr));
          ptrInc = IfThenElse::make(producedVals, ptrInc);
        } else if (emitAssemble) {
          // Emit code to resize idx (at result store loop nest)
          resizeIndices = IfThenElse::make(doResize, resizeIndices);
          ptrInc = Block::make({ptrInc, resizeIndices});
        }
        util::append(caseBody, {ptrInc});
      }
      cases.push_back({caseExpr, Block::make(caseBody)});
    }
    loopBody.push_back(needsMerge(lpLattice)
                       ? Case::make(cases, lpLattice.isFull())
                       : cases[0].second);

    // Emit code to conditionally increment sequential access ptr variables
    if (emitMerge) {
      loopBody.push_back(BlankLine::make());
      for (Iterator& iterator : lpIterators) {
        Expr ptr = iterator.getIteratorVar();
        Stmt inc = VarAssign::make(ptr, Add::make(ptr, 1));
        Expr tensorIdx = iterator.getIdxVar();
        Stmt maybeInc = (!iterator.isDense() && iterator.getIdxVar() != idx)
                        ? IfThenElse::make(Eq::make(tensorIdx, idx), inc) : inc;
        loopBody.push_back(maybeInc);
      }
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
      bool parallel = ctx.schedule.getAncestors(indexVar).size() == 1 &&
                      indexVar.isFree();
      for (size_t i = 0; i < ctx.schedule.getResultTensorPath().getSize(); i++){
        if (!ctx.iterators[resultPath.getStep(i)].isDense()) {
          parallel = false;
        }
      }
      Iterator iter = lpIterators[0];
      LoopKind loopKind = parallel ? LoopKind::Parallel : LoopKind::Serial;
      loop = For::make(iter.getIteratorVar(), iter.begin(), iter.end(), 1,
                       Block::make(loopBody), loopKind);
    }
    loops.push_back(loop);
  }
  util::append(code, loops);

  // Emit a store of the  segment size to the result ptr index
  // A.d2.ptr[A1_ptr + 1] = A2_ptr;
  if (emitAssemble && resultIterator.defined()) {
    Stmt ptrStore = resultIterator.storePtr();
    if (ptrStore.defined()) {
      util::append(code, {ptrStore});
    }
  }

//  code.push_back(Comment::make(util::fill("/"+toString(indexVar), '-', 70)));
  return code;
}

Stmt lower(TensorBase tensor, string funcName, set<Property> properties) {
  Context ctx;
  ctx.allocSize  = tensor.getAllocSize();
  ctx.properties = properties;

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

  // Initialize the result ptr variables
  TensorPath resultPath = ctx.schedule.getResultTensorPath();
  vector<Stmt> resultPtrInit;
  for (auto& indexVar : tensor.getIndexVars()) {
    Iterator iter = ctx.iterators[resultPath.getStep(indexVar)];
    if (iter.isSequentialAccess()) {
      Expr ptr = iter.getPtrVar();
      Expr ptrPrev = iter.getParent().getPtrVar();

      // Emit code to initialize the result ptr variable
      Stmt iteratorInit = VarAssign::make(iter.getPtrVar(), iter.begin(), true);
      resultPtrInit.push_back(iteratorInit);
    }
  }
  taco_iassert(results.size() == 1) << "An expression can only have one result";

  // Lower the iteration schedule
  vector<Stmt> code;
  auto& roots = ctx.schedule.getRoots();

  // Lower tensor expressions
  if (roots.size() > 0) {
    Iterator resultIterator = (resultPath.getSize() > 0)
        ? ctx.iterators[resultPath.getLastStep()]
        : ctx.iterators.getRoot(resultPath);  // e.g. `a = b(i) * c(i)`
    Target target;
    target.tensor = GetProperty::make(resultIterator.getTensor(),
                                      TensorProperty::Values);
    target.ptr = resultIterator.getPtrVar();

    for (auto& root : roots) {
      vector<Stmt> loopNest = lower(target, indexExpr, root, ctx);
      util::append(code, loopNest);
    }
  }
  // Lower scalar expressions
  else if (util::contains(properties,Compute)) {
    Expr expr = lowerToScalarExpression(indexExpr, ctx.iterators, ctx.schedule,
                                        map<TensorBase,ir::Expr>());
    TensorPath resultPath = ctx.schedule.getResultTensorPath();
    Expr resultTensorVar = ctx.iterators.getRoot(resultPath).getTensor();
    Expr vals = GetProperty::make(resultTensorVar, TensorProperty::Values);
    Stmt compute = Store::make(vals, 0, expr);
    code.push_back(compute);
  }

  // Create function
  vector<Stmt> body;
  body.push_back(Comment::make(tensor.getName() +
                "(" + util::join(vars) + ")" +
                               " = " + util::toString(indexExpr)));
  body.insert(body.end(), resultPtrInit.begin(), resultPtrInit.end());
  body.insert(body.end(), code.begin(), code.end());

  return Function::make(funcName, parameters, results, Block::make(body));
}
}}
