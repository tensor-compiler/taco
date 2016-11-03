#include "lower.h"

#include <vector>

#include "lower_scalar_expression.h"
#include "iterators.h"
#include "tensor_path.h"
#include "merge_rule.h"
#include "merge_lattice.h"
#include "iteration_schedule.h"

#include "internal_tensor.h"
#include "expr.h"
#include "operator.h"
#include "component_types.h"
#include "ir.h"
#include "var.h"
#include "storage/iterator.h"
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

vector<Stmt> lower(const set<Property>& properties,
                   const IterationSchedule& schedule,
                   const Iterators& iterators,
                   size_t level,
                   vector<Expr> indexVars,
                   map<Tensor,Expr> tensorVars);

/// Emit code to print the visited index variable coordinates
static vector<Stmt> printCoordinate(const vector<Expr>& indexVars) {
  vector<string> indexVarNames;
  indexVarNames.reserve((indexVars.size()));
  for (auto& indexVar : indexVars) {
    indexVarNames.push_back(util::toString(indexVar));
  }

  vector<string> fmtstrings(indexVars.size(), "%d");
  string format = util::join(fmtstrings, ",");
  vector<Expr> printvars = indexVars;
  return {Print::make("("+util::join(indexVarNames)+") = "  "("+format+")\\n",
                      printvars)};
}

Stmt initPtr(Expr ptr, Expr ptrPrev, Level levelFormat, Expr tensor) {
  Stmt initPtrStmt;
  switch (levelFormat.getType()) {
    case LevelType::Dense: {
      // Merging with dense formats should have been optimized away.
      ierror << "Doesn't make any sense to merge with a dense format.";
      break;
    }
    case LevelType::Sparse: {
      size_t dim = levelFormat.getDimension();
      Expr ptrArray = GetProperty::make(tensor, TensorProperty::Pointer, dim);
      Expr ptrVal = Load::make(ptrArray, ptrPrev);
      initPtrStmt = VarAssign::make(ptr, ptrVal);
      break;
    }
    case LevelType::Fixed: {
      not_supported_yet;
      break;
    }
  }
  iassert(initPtrStmt.defined());
  return initPtrStmt;
}

Expr exhausted(Expr ptr, Expr ptrPrev, Level levelFormat, Expr tensor) {
  Expr exhaustedExpr;
  switch (levelFormat.getType()) {
    case LevelType::Dense: {
      // Merging with dense formats should have been optimized away.
      ierror << "Doesn't make any sense to merge with a dense format.";
      break;
    }
    case LevelType::Sparse: {
      size_t dim = levelFormat.getDimension();
      Expr ptrArray = GetProperty::make(tensor, TensorProperty::Pointer, dim);
      Expr ptrVal = Load::make(ptrArray, Add::make(ptrPrev,1));
      exhaustedExpr = Lt::make(ptr, ptrVal);
      break;
    }
    case LevelType::Fixed: {
      not_supported_yet;
      break;
    }
  }
  iassert(exhaustedExpr.defined());
  return exhaustedExpr;
}

Stmt initTensorIdx(Expr tensorIdx, Expr ptr, Expr tensorVar,
                   TensorPathStep step) {
  Tensor tensor = step.getPath().getTensor();
  Level levelFormat = tensor.getFormat().getLevels()[step.getStep()];

  Stmt initTensorIndexStmt;
  switch (levelFormat.getType()) {
    case LevelType::Dense: {
      // Merging with dense formats should have been optimized away.
      ierror << "Doesn't make any sense to merge with a dense format.";
      break;
    }
    case LevelType::Sparse: {
      size_t dim = levelFormat.getDimension();
      Expr idxArray = GetProperty::make(tensorVar, TensorProperty::Index, dim);
      Expr idxVal = Load::make(idxArray, ptr);
      initTensorIndexStmt = VarAssign::make(tensorIdx, idxVal);
      break;
    }
    case LevelType::Fixed: {
      not_supported_yet;
      break;
    }
  }
  iassert(initTensorIndexStmt.defined());
  return initTensorIndexStmt;
}

Stmt initIdx(Expr idx, vector<Expr> tensorIndexVars) {
  return VarAssign::make(idx, Min::make(tensorIndexVars));
}

Stmt advance(Expr tensorIdx, Expr idx, Expr ptr) {
  Expr test    = Eq::make(tensorIdx, idx);
  Stmt incStmt = VarAssign::make(ptr, Add::make(ptr,1));
  return IfThenElse::make(test, incStmt);
}

static vector<Stmt> merge(size_t layer,
                          taco::Var var,
                          vector<Expr> indexVars,
                          const set<Property>& properties,
                          const IterationSchedule& schedule,
                          const Iterators& iterators,
                          const map<Tensor,Expr>& tensorVars) {

  MergeRule mergeRule = schedule.getMergeRule(var);
  MergeLattice mergeLattice = MergeLattice::make(mergeRule);
  vector<TensorPathStep> steps = mergeRule.getSteps();

  vector<Stmt> mergeLoops;

  TensorPathStep resultStep = mergeRule.getResultStep();
  Tensor resultTensor = schedule.getResultTensorPath().getTensor();
  Expr resultTensorVar = tensorVars.at(resultTensor);
  Expr resultPtr = iterators.getIterator(resultStep).getPtrVar();
  Expr resultPtrPrev = iterators.getPreviousIterator(resultStep).getPtrVar();

  // Emit code to initialize operand ptr variables
  for (auto& step : steps) {
    storage::Iterator iterator = iterators.getIterator(step);
    Expr ptr = iterator.getPtrVar();

    storage::Iterator iteratorPrev = iterators.getPreviousIterator(step);
    Expr ptrPrev = iteratorPrev.getPtrVar();

    Tensor tensor = step.getPath().getTensor();
    Expr tvar = tensorVars.at(tensor);

    Level levelFormat = tensor.getFormat().getLevels()[step.getStep()];
    Stmt initPtrStmt = initPtr(ptr, ptrPrev, levelFormat, tvar);
    mergeLoops.push_back(initPtrStmt);
  }

  // Emit one loop per lattice point lp
  auto latticePoints = mergeLattice.getPoints();
  for (size_t i=0; i < latticePoints.size(); ++i) {
    MergeLatticePoint lp = latticePoints[i];

    vector<Stmt> loopBody;
    vector<TensorPathStep> steps = lp.getSteps();

    // Emit code to initialize path index variables
    map<TensorPathStep, Expr> tensorIdxVariables;
    vector<Expr> tensorIdxVariablesVector;
    for (auto& step : steps) {
      Expr ptr = iterators.getIterator(step).getPtrVar();
      Tensor tensor = step.getPath().getTensor();
      Expr tvar = tensorVars.at(tensor);

      Expr stepIdx = iterators.getIterator(step).getIdxVar();
      tensorIdxVariables.insert({step, stepIdx});
      tensorIdxVariablesVector.push_back(stepIdx);

      Stmt initTensorIndexStmt = initTensorIdx(stepIdx, ptr, tvar, step);
      loopBody.push_back(initTensorIndexStmt);
    }

    // Iterate until any index has been exchaused
    Expr untilAnyExhausted;
    for (size_t i=0; i < steps.size(); ++i) {
      auto step = steps[i];
      Tensor tensor = step.getPath().getTensor();
      Expr ptr = iterators.getIterator(step).getPtrVar();
      Expr ptrPrev = iterators.getPreviousIterator(step).getPtrVar();;
      Level levelFormat = tensor.getFormat().getLevels()[step.getStep()];
      Expr tvar = tensorVars.at(tensor);

      Expr indexExhausted = exhausted(ptr, ptrPrev, levelFormat, tvar);
      untilAnyExhausted = (i == 0)
                          ? indexExhausted
                          : ir::And::make(untilAnyExhausted, indexExhausted);
    }

    // Emit code to initialize the index variable (min of path index variables)
    Expr idx = Var::make(var.getName(), typeOf<int>(), false);
    Stmt initIdxStmt = initIdx(idx, tensorIdxVariablesVector);
    loopBody.push_back(initIdxStmt);
    loopBody.push_back(BlankLine::make());

    // Emit a case per lattice point lq (non-strictly) dominated by lp
    auto dominatedPoints = mergeLattice.getDominatedPoints(lp);
    vector<pair<Expr,Stmt>> cases;
    for (MergeLatticePoint& lq : dominatedPoints) {
      auto steps = lq.getSteps();
      auto numLayers = schedule.numLayers();

      // Case expression
      Expr caseExpr;
      for (size_t i=0; i < steps.size(); ++i) {
        Expr caseTerm = Eq::make(tensorIdxVariables.at(steps[i]), idx);
        caseExpr = (i == 0) ? caseTerm : ir::And::make(caseExpr, caseTerm);
      }

      // Case body
      indexVars.push_back(idx);

      vector<Stmt> caseBody;

      // Print coordinate (only in base case)
      if (util::contains(properties, Print) && layer == numLayers-1) {
        auto print = printCoordinate(indexVars);
        util::append(caseBody, print);
      }

      // Recursive call to emit the next iteration schedule layer
      if (layer < numLayers-1) {
        auto nextLayer = lower(properties, schedule, iterators, layer+1,
                              indexVars, tensorVars);
        util::append(caseBody, nextLayer);
      }

      // Compute result values (only in base case)
      if (util::contains(properties, Compute) && layer == numLayers-1) {
        taco::Expr indexExpr = buildLatticePointExpression(schedule, lq);
        Expr computeExpr =
            lowerScalarExpression(indexExpr, iterators, schedule,  tensorVars);
        Expr vals = GetProperty::make(resultTensorVar, TensorProperty::Values);
        Stmt compute = Store::make(vals, resultPtr, computeExpr);
        util::append(caseBody, {compute});
      }

      // Insert into result tensor level idx
      // TODO: Move this ahead of the compute after increment the result
      //       iterator variable is done
      if (util::contains(properties, Assemble)) {
        if (util::contains(properties, Comment)) {
          Stmt comment = Comment::make("insert index value");
          util::append(caseBody, {BlankLine::make(), comment});
        }
        Expr idxArr = GetProperty::make(resultTensorVar,
                                        TensorProperty::Index, (int)layer);
        Stmt idxStore = Store::make(idxArr, resultPtr, idx);
        util::append(caseBody, {idxStore});
      }

      // Increment the results iterator variable
      Stmt ptrInc = VarAssign::make(resultPtr, Add::make(resultPtr, 1));
      if (layer < numLayers-1) {
        // If we didn't produce any values for the sub-tensor (the result ptr is
        // unchanged) then we don't insert an idx value.
        // TODO OPT: Only need to do this check if the merge rule intersects, as
        //           pure union rules will always produce values
        util::append(caseBody, {BlankLine::make()});
        storage::Iterator nextIterator = iterators.getNextIterator(resultStep);
        Expr ptrArr = GetProperty::make(resultTensorVar,
                                        TensorProperty::Pointer, layer+1);
        Expr producedVals = Gt::make(Load::make(ptrArr, Add::make(resultPtr,1)),
                                     Load::make(ptrArr, resultPtr));
        ptrInc =  IfThenElse::make(producedVals, ptrInc);
      }
      util::append(caseBody, {ptrInc});

      indexVars.pop_back();

      cases.push_back({caseExpr, Block::make(caseBody)});
    }
    Stmt casesStmt = Case::make(cases);
    loopBody.push_back(casesStmt);
    loopBody.push_back(BlankLine::make());

    // Emit code to conditionally increment ptr variables
    for (auto& step : steps) {
      Expr ptr = iterators.getIterator(step).getPtrVar();
      Expr tensorIdx = tensorIdxVariables.at(step);

      Stmt advanceStmt = advance(tensorIdx, idx, ptr);
      loopBody.push_back(advanceStmt);
    }

    mergeLoops.push_back(While::make(untilAnyExhausted, Block::make(loopBody)));

    if (i < latticePoints.size()-1) {
      mergeLoops.push_back(BlankLine::make());
    }
  }

  // Emit code to set result tensor level ptr
  if (util::contains(properties, Assemble)) {
    if (util::contains(properties, Comment)) {
      Stmt comment = Comment::make("set "+toString(resultTensorVar)+
                                   ".L"+to_string(layer)+".ptr");
      util::append(mergeLoops, {BlankLine::make(), comment});
    }

    Expr ptr = GetProperty::make(resultTensorVar,
                                 TensorProperty::Pointer, layer);
    Stmt ptrStore = Store::make(ptr, Add::make(resultPtrPrev,1), resultPtr);
    util::append(mergeLoops, {ptrStore});
  }

  return mergeLoops;
}

/// Lower one level of the iteration schedule. Dispatches to specialized lower
/// functions that recursively call this function to lower the next level
/// inside each loop at this level.
vector<Stmt> lower(const set<Property>& properties,
                   const IterationSchedule& schedule,
                   const Iterators& iterators,
                   size_t layer,
                   vector<Expr> indexVars,
                   map<Tensor,Expr> tensorVars) {
  vector<vector<taco::Var>> layers = schedule.getIndexVariables();
  iassert(layer < layers.size());


  vector<Stmt> levelCode;

  // Emit a loop sequence to merge the iteration space of incoming paths, and
  // recurse on the next layer in each loop.
  vector<taco::Var> vars = layers[layer];
  for (taco::Var var : vars) {

    // If there's only one incoming path then we emit a for loop.
    // Otherwise, we emit while loops that merge the incoming paths.
    vector<Stmt> loweredCode = merge(layer, var, indexVars, properties,
                                     schedule, iterators, tensorVars);
    util::append(levelCode, loweredCode);
  }

  return levelCode;
}

Stmt lower(const Tensor& tensor,
           string funcName, const set<Property>& properties) {
  string exprString = tensor.getName() +
                      "(" + util::join(tensor.getIndexVars()) + ")" +
                      " = " + util::toString(tensor.getExpr());

  // Pack the tensor and it's expression operands into the parameter list
  vector<Expr> parameters;
  vector<Expr> results;
  map<Tensor,Expr> tensorVars;

  // Pack result tensor into output parameter list
  Expr tensorVar = Var::make(tensor.getName(), typeOf<double>(),
                             tensor.getFormat());
  tensorVars.insert({tensor, tensorVar});
  parameters.push_back(tensorVar);

  // Pack operand tensors into input parameter list
  vector<Tensor> operands = internal::getOperands(tensor.getExpr());
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
    MergeRule mergeRule = schedule.getMergeRule(indexVar);

    TensorPathStep step = mergeRule.getResultStep();
    Tensor resultTensor = schedule.getResultTensorPath().getTensor();

    Expr tensorVar = tensorVars.at(resultTensor);
    Expr ptr = iterators.getIterator(step).getPtrVar();
    Expr ptrPrev = iterators.getPreviousIterator(step).getPtrVar();

    // Emit code to initialize the result ptr variable
    Level levelFormat = resultTensor.getFormat().getLevels()[step.getStep()];
    Stmt initPtrStmt = initPtr(ptr, ptrPrev, levelFormat, tensorVar);
    resultPtrInit.push_back(initPtrStmt);
  }

  // Lower the iteration schedule
  auto loweredCode = lower(properties, schedule, iterators, 0, {}, tensorVars);

  // Create function
  vector<Stmt> body;
  body.push_back(Comment::make(exprString));
  body.insert(body.end(), resultPtrInit.begin(), resultPtrInit.end());
  body.insert(body.end(), loweredCode.begin(), loweredCode.end());

  return Function::make(funcName, parameters, results, Block::make(body));
}

}}
