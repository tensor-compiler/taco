#include "lower.h"

#include <vector>

#include "lower_codegen.h"
#include "iterators.h"
#include "tensor_path.h"
#include "merge_lattice.h"
#include "iteration_schedule.h"
#include "available_exprs.h"

#include "tensor_base.h"
#include "expr.h"
#include "expr_rewriter.h"
#include "operator.h"
#include "component_types.h"
#include "var.h"
#include "ir/ir.h"
#include "ir/ir_visitor.h"
#include "ir/ir_codegen.h"
#include "storage/iterator.h"
#include "util/name_generator.h"
#include "util/collections.h"
#include "util/strings.h"

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
//  code.push_back(Comment::make(util::fill(toString(indexVar), '-', 70)));

  MergeLattice lattice = MergeLattice::make(indexExpr, indexVar, ctx.schedule);
  vector<Iterator> latticeIterators = ctx.iterators[lattice.getSteps()];

  TensorPath        resultPath      = ctx.schedule.getResultTensorPath();
  TensorPathStep    resultStep      = resultPath.getStep(indexVar);
  Iterator          resultIterator  = (resultStep.getPath().defined())
                                      ? ctx.iterators[resultStep]
                                      : Iterator();

  bool emitCompute  = util::contains(ctx.properties, Compute);
  bool emitAssemble = util::contains(ctx.properties, Assemble);

  bool merge = needsMerge(latticeIterators);
  bool reduceToVar = (indexVar.isReduction() &&
                      !ctx.schedule.hasFreeVariableDescendant(indexVar));

  // Emit code to initialize ptr variables: B2_ptr = B.d2.ptr[B1_ptr];
  if (merge) {
    for (auto& iterator : getSequentialAccessIterators(latticeIterators)) {
      Expr ptr = iterator.getPtrVar();
      Expr ptrPrev = iterator.getParent().getPtrVar();
      Expr tvar = iterator.getTensor();
      Expr iteratorVar = iterator.getIteratorVar();
      Stmt iteratorInit = VarAssign::make(iteratorVar, iterator.begin());
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
  for (MergeLatticePoint& lp : latticePoints) {
    vector<Stmt> loopBody;

    vector<Iterator> lpIterators = ctx.iterators[lp.getSteps()];
    bool emitCases = needsMerge(lpIterators);

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
    Expr idx = (emitCases)
               ? min(indexVar.getName(), sequentialAccessIterators, &loopBody)
               : lpIterators[0].getIdxVar();

    // Emit code to initialize random access ptr variables:
    // B2_ptr = (B1_ptr*3) + k;
    auto randomAccessIterators =
        getRandomAccessIterators(util::combine(lpIterators, {resultIterator}));
    for (Iterator& iterator : randomAccessIterators) {
      Expr val = ir::Add::make(ir::Mul::make(iterator.getParent().getPtrVar(),
                                             iterator.end()), idx);
      Stmt initPtr = VarAssign::make(iterator.getPtrVar(), val);
      loopBody.push_back(initPtr);
    }
    loopBody.push_back(BlankLine::make());

    // Emit one case per lattice point lq (non-strictly) dominated by lp
    auto dominatedPoints = lattice.getDominatedPoints(lp);
    vector<pair<Expr,Stmt>> cases;
    for (MergeLatticePoint& lq : dominatedPoints) {
      taco::Expr lqExpr = lq.getExpr();

      // Case expression
      vector<Expr> stepIdxEqIdx;
      vector<TensorPathStep> caseSteps = lq.simplify().getSteps();
      for (auto& caseStep : caseSteps) {
        Expr stepIdx = ctx.iterators[caseStep].getIdxVar();
        stepIdxEqIdx.push_back(Eq::make(stepIdx, idx));
      }
      Expr caseExpr = conjunction(stepIdxEqIdx);

      // Case body
      vector<Stmt> caseBody;

      // Emit compute code. Cases: ABOVE_LAST_FREE, LAST_FREE or BELOW_LAST_FREE
      ComputeCase computeCase = getComputeCase(indexVar, ctx.schedule);

      // Emit available sub-expressions at this level
      if (ABOVE_LAST_FREE == computeCase && emitCompute) {
        vector<taco::Var> visited = ctx.schedule.getAncestors(indexVar);
        vector<taco::Expr> availExprs = getAvailableExpressions(lqExpr,visited);

        map<taco::Expr,taco::Expr> substitutions;
        for (const taco::Expr& availExpr : availExprs) {
          // If it's an expression we've emitted (in a higher loop) we ignore it
          if (isa<Read>(availExpr) &&
              util::contains(ctx.temporaries,to<Read>(availExpr).getTensor())) {
            continue;
          }

          std::string name = util::uniqueName("t");
          TensorBase t(name, ComponentType::Double);
          substitutions.insert({availExpr, taco::Read(t)});

          Expr tensorVar = Var::make(name, typeOf<double>(), false);
          ctx.temporaries.insert({t, tensorVar});

          Expr availIRExpr = lowerToScalarExpression(availExpr, ctx.iterators,
                                                     ctx.schedule,
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

      /// Compute and add available expressions to a reduction variable
      if (BELOW_LAST_FREE == computeCase && emitCompute) {
        Expr scalarexpr = lowerToScalarExpression(lqExpr, ctx.iterators,
                                                  ctx.schedule,
                                                  ctx.temporaries);

        iassert(reduceToVar);
        caseBody.push_back(compoundAssign(reductionVar, scalarexpr));
      }

      // Compute and store available expression to results
      if (LAST_FREE == computeCase && emitCompute) {
        Expr scalarexpr = lowerToScalarExpression(lqExpr, ctx.iterators,
                                                  ctx.schedule,
                                                  ctx.temporaries);
        Expr resultPtr = resultIterator.getPtrVar();
        Expr vals = GetProperty::make(resultIterator.getTensor(),
                                      TensorProperty::Values);

        // Store to result tensor
        Stmt storeResult = ctx.schedule.hasReductionVariableAncestor(indexVar)
            ? compoundStore(vals, resultPtr, scalarexpr)
            : Store::make(vals, resultPtr, scalarexpr);
//        caseBody.push_back(Comment::make(toString(storeResult)));
      }

      // Emit code to compute result values in base case (DEPRECATED)
      if (emitCompute) {
        if (ctx.schedule.getChildren(indexVar).size() == 0) {
          Iterator resultIterator = (resultPath.getSize() > 0)
                                    ? ctx.iterators[resultPath.getLastStep()]
                                    : ctx.iterators.getRoot(resultPath);
          Expr resultPtr = resultIterator.getPtrVar();
          Expr scalarExpr = lowerToScalarExpression(lq.getExpr(), ctx.iterators,
                                                    ctx.schedule,
                                                    ctx.temporaries);
          Expr vals = GetProperty::make(resultIterator.getTensor(),
                                        TensorProperty::Values);
          Stmt compute = compoundStore(vals, resultPtr, scalarExpr);
          util::append(caseBody, {compute});
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
    loopBody.push_back(!emitCases ? cases[0].second : Case::make(cases));

    // Emit code to conditionally increment sequential access ptr variables
    if (merge) {
      loopBody.push_back(BlankLine::make());
      for (Iterator& iterator : sequentialAccessIterators) {
        Expr ptr = iterator.getIteratorVar();
        Stmt inc = VarAssign::make(ptr, Add::make(ptr, 1));
        Expr tensorIdx = iterator.getIdxVar();
        Stmt maybeInc = (emitCases)
                        ? IfThenElse::make(Eq::make(tensorIdx, idx), inc) : inc;
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
        Iterator iter = ctx.iterators[mergeStep];
        stepIterLqEnd.push_back(Lt::make(iter.getIteratorVar(), iter.end()));
      }
      Expr untilAnyExhausted = conjunction(stepIterLqEnd);
      loop = While::make(untilAnyExhausted, Block::make(loopBody));
    }
    else {
      iassert(lp.simplify().getSteps().size() == 1);
      Iterator iter = lpIterators[0];
      loop = For::make(iter.getIteratorVar(), iter.begin(), iter.end(), 1,
                       Block::make(loopBody));
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

Stmt lower(const TensorBase& tensor, string funcName,
           const set<Property>& properties) {
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

  // Pack result tensor into output parameter list
  Expr tensorVar = Var::make(name, typeOf<double>(), tensor.getFormat());
  tensorVars.insert({tensor, tensorVar});
  results.push_back(tensorVar);

  // Pack operand tensors into input parameter list
  vector<TensorBase> operands = internal::getOperands(indexExpr);
  for (TensorBase& operand : operands) {
    iassert(!util::contains(tensorVars, operand));
    Expr operandVar = Var::make(operand.getName(), typeOf<double>(),
                                operand.getFormat());
    tensorVars.insert({operand, operandVar});
    parameters.push_back(operandVar);
  }

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
      Stmt iteratorInit = VarAssign::make(iter.getPtrVar(), iter.begin());
      resultPtrInit.push_back(iteratorInit);
    }
  }

  // Lower the iteration schedule
  vector<Stmt> code;
  auto& roots = ctx.schedule.getRoots();

  // Lower scalar expressions
  if (roots.size() == 0 && util::contains(properties,Compute)) {
    Expr expr = lowerToScalarExpression(indexExpr, ctx.iterators, ctx.schedule,
                                        map<TensorBase,ir::Expr>());
    TensorPath resultPath = ctx.schedule.getResultTensorPath();
    Expr resultTensorVar = ctx.iterators.getRoot(resultPath).getTensor();
    Expr vals = GetProperty::make(resultTensorVar, TensorProperty::Values);
    Stmt compute = Store::make(vals, 0, expr);
    code.push_back(compute);
  }
  // Lower tensor expressions
  else {
    for (auto& root : roots) {
      vector<Stmt> loopNest = lower(indexExpr, root, ctx);
      util::append(code, loopNest);
    }
  }

  // Create function
  vector<Stmt> body;
//  body.push_back(Comment::make(tensor.getName() +
//                "(" + util::join(vars) + ")" +
//                               " = " + util::toString(indexExpr)));
  body.insert(body.end(), resultPtrInit.begin(), resultPtrInit.end());
  body.insert(body.end(), code.begin(), code.end());

  return Function::make(funcName, parameters, results, Block::make(body));
}
}}
