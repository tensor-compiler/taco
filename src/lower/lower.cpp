#include "lower.h"

#include <vector>

#include "lower_codegen.h"
#include "iterators.h"
#include "tensor_path.h"
#include "merge_lattice.h"
#include "iteration_schedule.h"
#include "available_exprs.h"

#include "internal_tensor.h"
#include "expr.h"
#include "expr_rewriter.h"
#include "operator.h"
#include "component_types.h"
#include "ir.h"
#include "ir_visitor.h"
#include "ir_codegen.h"
#include "var.h"
#include "storage/iterator.h"
#include "util/name_generator.h"
#include "util/collections.h"
#include "util/strings.h"

using namespace std;

namespace taco {
namespace lower {

using namespace taco::ir;

using taco::internal::Tensor;
using taco::ir::Expr;
using taco::ir::Var;
using taco::ir::Add;

struct Context {
  /// Determines what kind of code to emit (e.g. compute and/or assembly)
  const set<Property>&     properties;

  /// The iteration schedule to use for lowering the index expression
  const IterationSchedule& schedule;

  /// The iterators of the tensor tree levels
  const Iterators&         iterators;

  /// The size of initial memory allocations
  const size_t             allocSize;

  /// Maps tensors to IR variables
  const map<Tensor,Expr>&  tensorVars;

  /// Maps tensor (scalar) temporaries to IR variables.
  /// (Not clear if this approach to temporaries is too hacky.)
  map<Tensor,Expr>         temporaries;

  /// Constructor
  Context(const set<Property>&     properties,
          const IterationSchedule& schedule,
          const Iterators&         iterators,
          map<Tensor,Expr>&        tensorVars,
          const size_t             allocSize)
      : properties(properties), schedule(schedule), iterators(iterators),
        allocSize(allocSize), tensorVars(tensorVars) {}
};

// The steps of a merge rule must be merged iff two or more of them are dense.
static bool needsMerge(vector<TensorPathStep> mergedSteps) {
  int sparseOperands = 0;
  for (auto& step : mergedSteps) {
    Format format = step.getPath().getTensor().getFormat();
    if (format.getLevels()[step.getStep()].getType() != LevelType::Dense) {
      sparseOperands++;
    }
  }
  return (sparseOperands > 1);
}

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

static vector<Stmt> lower(const taco::Expr& indexExpr,
                          const taco::Var&  indexVar,
                          Context&          ctx) {
  vector<Stmt> code;
  code.push_back(BlankLine::make());
  code.push_back(Comment::make(util::fill(toString(indexVar), '-', 70)));

  MergeLattice lattice = MergeLattice::make(indexExpr, indexVar, ctx.schedule);
  vector<TensorPathStep> latticeSteps = lattice.getSteps();

  TensorPath        resultPath      = ctx.schedule.getResultTensorPath();
  TensorPathStep    resultStep      = resultPath.getStep(indexVar);
  Tensor            resultTensor    = ctx.schedule.getTensor();
  Expr              resultTensorVar = ctx.tensorVars.at(resultTensor);
  storage::Iterator resultIterator  = (resultStep.getPath().defined())
                                      ? ctx.iterators.getIterator(resultStep)
                                      : storage::Iterator();

  bool merge = needsMerge(latticeSteps);
  bool reduceToVar = (indexVar.isReduction() &&
                      !ctx.schedule.hasFreeVariableDescendant(indexVar));

  // Emit code to initialize ptr variables: B2_ptr = B.d2.ptr[B1_ptr];
  if (merge) {
    for (auto& step : latticeSteps) {
      storage::Iterator iter = ctx.iterators.getIterator(step);
      storage::Iterator iterPrev = ctx.iterators.getPreviousIterator(step);

      Expr ptr = iter.getPtrVar();
      Expr ptrPrev = iterPrev.getPtrVar();
      Tensor tensor = step.getPath().getTensor();
      Expr tvar = ctx.tensorVars.at(tensor);
      Expr iteratorVar = iter.getIteratorVar();
      Stmt iteratorInit = VarAssign::make(iteratorVar, iter.begin());
      code.push_back(iteratorInit);
    }
  }

  // Emit code to initialize reduction variable: tk = 0.0;
  Expr reductionVar;
  if (reduceToVar) {
    reductionVar = Var::make("t"+indexVar.getName(), typeOf<double>(), false);
    Stmt reductionVarInit = VarAssign::make(reductionVar, 0.0);
    util::append(code, {reductionVarInit});
  }

  // Emit one loop per lattice point lp
  vector<Stmt> loops;
  auto latticePoints = lattice.getPoints();
  if (!merge) latticePoints = {latticePoints[0]};  // TODO: Get rid of this
  for (MergeLatticePoint lp : latticePoints) {
    vector<Stmt> loopBody;

    vector<TensorPathStep> lpSteps = lp.getSteps();

    // Collect all the tensor idx variables (ia, iB, jC, ...)
    map<TensorPathStep, Expr> tensorIdxVariables;
    for (TensorPathStep& step : lpSteps) {
      Expr stepIdx = ctx.iterators.getIterator(step).getIdxVar();
      tensorIdxVariables.insert({step, stepIdx});
    }

    // Emit code to initialize sequential access idx variables:
    // kB = B.d2.idx[B2_ptr];
    auto sequentialAccessSteps = getSequentialAccessSteps(lpSteps);
    vector<Expr> mergeIdxVariables;
    for (TensorPathStep& step : sequentialAccessSteps) {
      storage::Iterator iterator = ctx.iterators.getIterator(step);
      Stmt initIdx = iterator.initDerivedVar();
      loopBody.push_back(initIdx);
      mergeIdxVariables.push_back(iterator.getIdxVar());
    }

    // Emit code to initialize the index variable: k = min(kB, kc);
    Expr idx ;
    if (merge) {
      idx = Var::make(indexVar.getName(), typeOf<int>(), false);
      Stmt initIdxStmt = mergePathIndexVars(idx, mergeIdxVariables);
      loopBody.push_back(initIdxStmt);
    }
    else {
      idx = ctx.iterators.getIterator(lpSteps[0]).getIdxVar();
      const_cast<Var*>(idx.as<Var>())->name = indexVar.getName();
    }

    // Emit code to initialize random access ptr variables:
    // B2_ptr = (B1_ptr*3) + k;
    auto randomAccessSteps = getRandomAccessSteps(lpSteps);
    if (resultIterator.defined() && resultIterator.isRandomAccess()) {
      randomAccessSteps.push_back(resultStep);  // include the result ptr var
    }
    for (TensorPathStep& step : randomAccessSteps) {
      storage::Iterator iterPrev = ctx.iterators.getPreviousIterator(step);
      storage::Iterator iter = ctx.iterators.getIterator(step);
      Expr ptrVal = ir::Add::make(ir::Mul::make(iterPrev.getPtrVar(),
                                                iter.end()), idx);
      Stmt initPtr = VarAssign::make(iter.getPtrVar(), ptrVal);
      loopBody.push_back(initPtr);
    }

    // Emit one case per lattice point lq (non-strictly) dominated by lp
    auto dominatedPoints = lattice.getDominatedPoints(lp);
    vector<pair<Expr,Stmt>> cases;
    for (MergeLatticePoint& lq : dominatedPoints) {
      vector<TensorPathStep> lqSteps = lq.getSteps();
      taco::Expr lqExpr = lq.getExpr();

      // Case expression
      vector<Expr> stepIdxEqIdx;
      vector<TensorPathStep> caseSteps = lq.simplify().getSteps();
      for (auto& caseStep : caseSteps) {
        stepIdxEqIdx.push_back(Eq::make(tensorIdxVariables.at(caseStep), idx));
      }
      Expr caseExpr = conjunction(stepIdxEqIdx);

      // Case body
      vector<Stmt> caseBody;

      // Emit compute code. Cases: ABOVE_LAST_FREE, LAST_FREE or BELOW_LAST_FREE
      ComputeCase computeCase = getComputeCase(indexVar, ctx.schedule);

      // Emit available sub-expressions at this level
      if (computeCase == ABOVE_LAST_FREE &&
          util::contains(ctx.properties, Compute)) {

        vector<taco::Var> visited = ctx.schedule.getAncestors(indexVar);
        vector<taco::Expr> availExprs = getAvailableExpressions(lqExpr,visited);

        caseBody.push_back(BlankLine::make());
        caseBody.push_back(Comment::make("Emit available sub-expressions"));

        map<taco::Expr,taco::Expr> substitutions;
        for (auto& availExpr : availExprs) {
          std::string name = util::uniqueName("t");

          internal::Tensor t(name, ComponentType::Double);
          substitutions.insert({availExpr, taco::Read(t)});

          Expr tensorVar = Var::make(name, typeOf<double>(), false);
          ctx.temporaries.insert({t, tensorVar});

          Expr availIRExpr = lowerToScalarExpression(availExpr, ctx.iterators,
                                                     ctx.schedule,
                                                     ctx.tensorVars,
                                                     ctx.temporaries);
          caseBody.push_back(VarAssign::make(tensorVar, availIRExpr));
        }
        lqExpr = internal::replace(lqExpr, substitutions);
      }

      // Recursive call to emit iteration schedule children
      for (auto& child : ctx.schedule.getChildren(indexVar)) {
        auto childCode = lower(lqExpr, child, ctx);
        util::append(caseBody, childCode);
      }

      // Compute and store available expression
      if ((computeCase == LAST_FREE || computeCase == BELOW_LAST_FREE) &&
          util::contains(ctx.properties, Compute)) {

        Expr scalarexpr = lowerToScalarExpression(lqExpr, ctx.iterators,
                                                  ctx.schedule, ctx.tensorVars,
                                                  ctx.temporaries);

        // Store result to result tensor (last free) or reduction variable
        // (below last free)
        if (computeCase == LAST_FREE) {
          auto resultPath = ctx.schedule.getResultTensorPath();
          Expr resultPtr = resultIterator.getPtrVar();
          Expr vals = GetProperty::make(resultTensorVar,TensorProperty::Values);

          // Store to result tensor
          Stmt storeResult = ctx.schedule.hasReductionVariableAncestor(indexVar)
              ? compoundStore(vals, resultPtr, scalarexpr)
              : Store::make(vals, resultPtr, scalarexpr);
//           caseBody.push_back(storeResult);
//          caseBody.push_back(Comment::make(toString(storeResult)));
        }
        else if (computeCase == BELOW_LAST_FREE) {
          iassert(reduceToVar);
          caseBody.push_back(compoundAssign(reductionVar, scalarexpr));
        }
      }

      // Emit code to compute result values in base case (DEPRECATED)
      if (util::contains(ctx.properties, Compute)) {
        if (ctx.schedule.getChildren(indexVar).size() == 0) {

          auto resultPath = ctx.schedule.getResultTensorPath();
          storage::Iterator resultIterator = (resultTensor.getOrder() > 0)
              ? ctx.iterators.getIterator(resultPath.getLastStep())
              : ctx.iterators.getRootIterator();
          Expr resultPtr = resultIterator.getPtrVar();

          Expr scalarExpr = lowerToScalarExpression(lq.getExpr(), ctx.iterators,
                                                    ctx.schedule, ctx.tensorVars,
                                                    ctx.temporaries);

          Expr vals = GetProperty::make(resultTensorVar,TensorProperty::Values);
          Stmt compute = compoundStore(vals, resultPtr, scalarExpr);
          util::append(caseBody, {compute});
        }
      }

      // Emit a store of the index variable value to the result idx index array
      // A.d2.idx[A2_ptr] = j;
      if (util::contains(ctx.properties, Assemble) && resultIterator.defined()){
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

        if (resultStep != resultStep.getPath().getLastStep()) {
          util::append(caseBody, {BlankLine::make()});
          storage::Iterator iterNext =ctx.iterators.getNextIterator(resultStep);

          // Emit code to resize idx and ptr
          if (util::contains(ctx.properties, Assemble)) {
            Stmt resizePtr = iterNext.resizePtrStorage(newSize);
            resizeIndices = Block::make({resizePtr, resizeIndices});
            resizeIndices = IfThenElse::make(doResize, resizeIndices);
            ptrInc = Block::make({ptrInc, resizeIndices});
          }

          Expr ptrArr = GetProperty::make(resultTensorVar,
                                          TensorProperty::Pointer,
                                          resultStep.getStep()+1);
          Expr producedVals =
              Gt::make(Load::make(ptrArr, Add::make(resultPtr,1)),
                       Load::make(ptrArr, resultPtr));
          ptrInc = IfThenElse::make(producedVals, ptrInc);
        } else if (util::contains(ctx.properties, Assemble)) {
          // Emit code to resize idx (at result store loop nest)
          resizeIndices = IfThenElse::make(doResize, resizeIndices);
          ptrInc = Block::make({ptrInc, resizeIndices});
        }
        util::append(caseBody, {ptrInc});
      }
      cases.push_back({caseExpr, Block::make(caseBody)});
    }
    loopBody.push_back(!merge ? cases[0].second : Case::make(cases));
    loopBody.push_back(BlankLine::make());

    // Emit code to conditionally increment sequential access ptr variables
    if (merge) {
      for (auto& step : sequentialAccessSteps) {
        Expr ptr = ctx.iterators.getIterator(step).getIteratorVar();
        Stmt inc = VarAssign::make(ptr, Add::make(ptr, 1));
        Expr tensorIdx = tensorIdxVariables.at(step);
        Stmt maybeInc = IfThenElse::make(Eq::make(tensorIdx, idx), inc);
        loopBody.push_back(maybeInc);
      }
    }

    // Emit loop (while loop for merges and for loop for non-merges)
    Stmt loop;
    if (merge) {
      // Loop until any index has been exchaused
      vector<Expr> stepIterLqEnd;
      vector<TensorPathStep> mergeSteps = lp.simplify().getSteps();
      for (auto& mergeStep : mergeSteps) {
        storage::Iterator iter = ctx.iterators.getIterator(mergeStep);
        stepIterLqEnd.push_back(Lt::make(iter.getIteratorVar(), iter.end()));
      }
      Expr untilAnyExhausted = conjunction(stepIterLqEnd);
      loop = While::make(untilAnyExhausted, Block::make(loopBody));
    }
    else {
      iassert(lp.simplify().getSteps().size() == 1);
      storage::Iterator iter = ctx.iterators.getIterator(lpSteps[0]);
      loop = For::make(iter.getIteratorVar(), iter.begin(), iter.end(), 1,
                       Block::make(loopBody));
    }
    loops.push_back(loop);
    loops.push_back(BlankLine::make());
  }
  util::append(code, loops);

  // Emit a store of the  segment size to the result ptr index
  // A.d2.ptr[A1_ptr + 1] = A2_ptr;
  if (util::contains(ctx.properties, Assemble) && resultIterator.defined()) {
    Stmt ptrStore = resultIterator.storePtr();
    if (ptrStore.defined()) {
      util::append(code, {ptrStore});
    }
  }

  code.push_back(Comment::make(util::fill("/"+toString(indexVar), '-', 70)));
  code.push_back(BlankLine::make());
  return code;
}

Stmt lower(const Tensor& tensor, string funcName,
           const set<Property>& properties) {
  auto name = tensor.getName();
  auto vars = tensor.getIndexVars();
  auto indexExpr = tensor.getExpr();
  auto allocSize = tensor.getAllocSize();

  string exprString = name + "(" + util::join(vars) + ")" +
                      " = " + util::toString(indexExpr);

  // Pack the tensor and it's expression operands into the parameter list
  vector<Expr> parameters;
  vector<Expr> results;
  map<Tensor,Expr> tensorVars;

  // Pack result tensor into output parameter list
  Expr tensorVar = Var::make(name, typeOf<double>(), tensor.getFormat());
  tensorVars.insert({tensor, tensorVar});
  results.push_back(tensorVar);

  // Pack operand tensors into input parameter list
  vector<Tensor> operands = internal::getOperands(indexExpr);
  for (auto& operand : operands) {
    iassert(!util::contains(tensorVars, operand));
    Expr operandVar = Var::make(operand.getName(), typeOf<double>(),
                                operand.getFormat());
    tensorVars.insert({operand, operandVar});
    parameters.push_back(operandVar);
  }

  // Create the schedule and the iterators of the lowered code
  IterationSchedule schedule = IterationSchedule::make(tensor);
  Iterators iterators(schedule, tensorVars);

  // Initialize the result ptr variables
  vector<Stmt> resultPtrInit;
  for (auto& indexVar : tensor.getIndexVars()) {
    TensorPath     resultPath = schedule.getResultTensorPath();
    TensorPathStep resultStep = resultPath.getStep(indexVar);
    Tensor         result     = schedule.getResultTensorPath().getTensor();

    Expr tensorVar = tensorVars.at(result);
    storage::Iterator iterator = iterators.getIterator(resultStep);
    Expr ptr = iterator.getPtrVar();
    Expr ptrPrev = iterators.getPreviousIterator(resultStep).getPtrVar();

    // Emit code to initialize the result ptr variable
    Stmt iteratorInit = VarAssign::make(iterator.getPtrVar(), iterator.begin());
    resultPtrInit.push_back(iteratorInit);
  }

  // Lower the iteration schedule
  vector<Stmt> code;
  auto& roots = schedule.getRoots();

  // Lower scalar expressions
  if (roots.size() == 0 && util::contains(properties, Compute)) {
    Expr expr = lowerToScalarExpression(indexExpr,iterators,schedule,tensorVars,
                                        map<internal::Tensor,ir::Expr>());
    Expr resultTensorVar = tensorVars.at(schedule.getTensor());
    Expr vals = GetProperty::make(resultTensorVar, TensorProperty::Values);
    Stmt compute = Store::make(vals, 0, expr);
    code.push_back(compute);
  }
  // Lower tensor expressions
  else {
    Context ctx(properties, schedule, iterators, tensorVars, allocSize);
    for (auto& root : roots) {
      vector<Stmt> loopNest = lower(indexExpr, root, ctx);
      util::append(code, loopNest);
    }
  }

  // Create function
  vector<Stmt> body;
  body.push_back(Comment::make(exprString));
  body.insert(body.end(), resultPtrInit.begin(), resultPtrInit.end());
  body.insert(body.end(), code.begin(), code.end());

  return Function::make(funcName, parameters, results, Block::make(body));
}
}}
