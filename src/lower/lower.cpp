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
#include "ir_visitor.h"
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

vector<Stmt> lower(const Expr& expr,
                   const set<Property>& properties,
                   const IterationSchedule& schedule,
                   const Iterators& iterators,
                   const taco::Expr& indexExpr,
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

Stmt initIdx(Expr idx, vector<Expr> tensorIndexVars);
Stmt initIdx(Expr idx, vector<Expr> tensorIndexVars) {
  return VarAssign::make(idx, Min::make(tensorIndexVars));
}

static vector<Stmt> merge(const Expr& expr,
                          const taco::Expr& indexExpr,
                          size_t layer,
                          taco::Var var,
                          vector<Expr> indexVars,
                          const set<Property>& properties,
                          const IterationSchedule& schedule,
                          const Iterators& iterators,
                          const map<Tensor,Expr>& tensorVars) {

  MergeRule mergeRule = schedule.getMergeRule(var);
  MergeLattice mergeLattice = MergeLattice::make(mergeRule);
  vector<TensorPathStep> mergeRuleSteps = mergeRule.getSteps();

  vector<Stmt> mergeCode;

  TensorPathStep resultStep = mergeRule.getResultStep();
  storage::Iterator resultIterator = (resultStep.getPath().defined())
                                     ? iterators.getIterator(resultStep)
                                     : storage::Iterator();

  Tensor resultTensor = schedule.getTensor();
  Expr resultTensorVar = tensorVars.at(resultTensor);

  // Turn of merging if there's one or zero sparse arguments
  int sparseOperands = 0;
  for (auto& step : mergeRuleSteps) {
    Format format = step.getPath().getTensor().getFormat();
    if (format.getLevels()[step.getStep()].getType() == LevelType::Sparse) {
      sparseOperands++;
    }
  }
  bool noMerge = (sparseOperands <= 1);

  // Emit code to initialize iterator variables
  if (!noMerge) {
    for (auto& step : mergeRuleSteps) {
      storage::Iterator iterator = iterators.getIterator(step);
      Expr ptr = iterator.getPtrVar();

      storage::Iterator iteratorPrev = iterators.getPreviousIterator(step);
      Expr ptrPrev = iteratorPrev.getPtrVar();

      Tensor tensor = step.getPath().getTensor();
      Expr tvar = tensorVars.at(tensor);

      Expr iteratorVar = iterator.getIteratorVar();
      Stmt iteratorInit = VarAssign::make(iteratorVar, iterator.begin());
      mergeCode.push_back(iteratorInit);
    }
  }

  // Emit one loop per lattice point lp
  vector<Stmt> mergeLoops;
  auto latticePoints = mergeLattice.getPoints();
  for (size_t i=0; i < latticePoints.size(); ++i) {
    MergeLatticePoint lp = latticePoints[i];
    MergeLatticePoint lpSimplified = simplify(lp);

    vector<Stmt> loopBody;
    vector<TensorPathStep> lpSteps = lp.getSteps();
    vector<TensorPathStep> lpSimplifiedSteps = lpSimplified.getSteps();

    // Emit code to initialize sparse idx variables
    map<TensorPathStep, Expr> tensorIdxVariables;
    vector<Expr> tensorIdxVariablesVector;
    for (auto& step : lpSteps) {
      Format format = step.getPath().getTensor().getFormat();
      if (format.getLevels()[step.getStep()].getType() == LevelType::Sparse) {
        storage::Iterator iterator = iterators.getIterator(step);

        Expr idxStep = iterator.getIdxVar();
        tensorIdxVariables.insert({step, idxStep});
        tensorIdxVariablesVector.push_back(idxStep);

        Stmt initDerivedVars = iterator.initDerivedVar();
        loopBody.push_back(initDerivedVars);
      }
    }

    if (tensorIdxVariablesVector.size() == 0) {
      iassert(lpSteps.size() > 0);
      Expr idxStep = iterators.getIterator(lpSteps[0]).getIdxVar();
      tensorIdxVariablesVector.push_back(idxStep);
    }

    // Emit code to initialize the index variable (min of path index variables)
    Expr idx;
    if (noMerge) {
      idx = tensorIdxVariablesVector[0];
      const_cast<Var*>(idx.as<Var>())->name = var.getName();
    }
    else {
      idx = Var::make(var.getName(), typeOf<int>(), false);
      Stmt initIdxStmt = initIdx(idx, tensorIdxVariablesVector);
      loopBody.push_back(initIdxStmt);
    }

    // Emit code to initialize dense ptr variables
    for (auto& step : lpSteps) {
      Format format = step.getPath().getTensor().getFormat();
      if (format.getLevels()[step.getStep()].getType() == LevelType::Dense) {
        storage::Iterator iterator = iterators.getIterator(step);
        storage::Iterator iteratorPrev = iterators.getPreviousIterator(step);

        Expr stepIdx = iterator.getIdxVar();
        tensorIdxVariables.insert({step, stepIdx});

        Expr ptrVal = ir::Add::make(ir::Mul::make(iteratorPrev.getPtrVar(),
                                                  iterator.end()), idx);
        Stmt initPtr = VarAssign::make(iterator.getPtrVar(), ptrVal);
        loopBody.push_back(initPtr);
      }
    }
    if (resultIterator.defined() && resultIterator.isRandomAccess()) {
      auto resultPrevIterator = iterators.getPreviousIterator(resultStep);
      Expr ptrVal = ir::Add::make(ir::Mul::make(resultPrevIterator.getPtrVar(),
                                                resultIterator.end()), idx);
      Stmt initResultPtr = VarAssign::make(resultIterator.getPtrVar(), ptrVal);
      loopBody.push_back(initResultPtr);
    }
    loopBody.push_back(BlankLine::make());

    // Emit one case per lattice point lq (non-strictly) dominated by lp
    auto dominatedPoints = mergeLattice.getDominatedPoints(lp);
    vector<pair<Expr,Stmt>> cases;
    for (MergeLatticePoint& lq : dominatedPoints) {
      MergeLatticePoint lqSimplified = simplify(lq);

      auto lqSteps = lq.getSteps();
      auto lqSimplifiedSteps = lqSimplified.getSteps();

      auto numLayers = schedule.numLayers();

      // Case expression
      Expr caseExpr;
      for (size_t i=0; i < lqSimplifiedSteps.size(); ++i) {
        Expr caseTerm = Eq::make(tensorIdxVariables.at(lqSteps[i]), idx);
        caseExpr = (i == 0) ? caseTerm : ir::And::make(caseExpr, caseTerm);
      }

      // Case body
      vector<Stmt> caseBody;
      indexVars.push_back(idx);

      // Print coordinate (only in base case)
      if (util::contains(properties, Print) && layer == numLayers-1) {
        auto print = printCoordinate(indexVars);
        util::append(caseBody, print);
      }


      // Emit code to compute result values (only in base case)
      if (util::contains(properties, Compute) && layer == numLayers-1) {
        storage::Iterator resultIterator =
            (resultTensor.getOrder() > 0)
            ?iterators.getIterator(schedule.getResultTensorPath().getLastStep())
            :iterators.getRootIterator();
        Expr resultPtr = resultIterator.getPtrVar();

        // Build the index expression for this case
        Expr lqExpr = removeExpressions(expr, lqSteps, iterators);


        Expr vals = GetProperty::make(resultTensorVar, TensorProperty::Values);

        switch (var.getKind()) {
          case taco::Var::Free:
            // Do nothing
            break;
          case taco::Var::Sum:
            lqExpr = Add::make(Load::make(vals, resultPtr), lqExpr);
            break;
        }

        Stmt compute = Store::make(vals, resultPtr, lqExpr);
        util::append(caseBody, {compute});
      }
      else {
        // We need to compute subexpressions when they become available, so that
        // we have them available at later levels. It is a matter of efficiency,
        // but it's also a matter of avoiding buildLatticePointExpression
        // removing expressions from earlier in the iteration schedule.
//        buildLatticePointExpression(schedule, lq);
      }

      // Recursive call to emit the next iteration schedule layer
      if (layer < numLayers-1) {
        auto nextLayer = lower(expr, properties, schedule, iterators,
                               indexExpr, layer+1, indexVars, tensorVars);
        util::append(caseBody, nextLayer);
      }

      // Emit code to store the index variable value to idx
      if (util::contains(properties, Assemble) && resultIterator.defined()) {
        Stmt idxStore = resultIterator.storeIdx(idx);
        if (idxStore.defined()) {
          if (util::contains(properties, Comment)) {
            Stmt comment = Comment::make("insert index value");
            util::append(caseBody, {BlankLine::make(), comment});
          }
          util::append(caseBody, {idxStore});
        }
      }

      // Emit code to increment the results iterator variable
      if (resultIterator.defined() && !resultIterator.isRandomAccess()) {
        Expr resultPtr = resultIterator.getPtrVar();

        Stmt ptrInc = VarAssign::make(resultPtr, Add::make(resultPtr, 1));
        if (resultStep != resultStep.getPath().getLastStep()) {
          util::append(caseBody, {BlankLine::make()});
          storage::Iterator nextIterator= iterators.getNextIterator(resultStep);
          Expr ptrArr = GetProperty::make(resultTensorVar,
                                          TensorProperty::Pointer, layer+1);
          Expr producedVals =
              Gt::make(Load::make(ptrArr, Add::make(resultPtr,1)),
                       Load::make(ptrArr, resultPtr));
          ptrInc =  IfThenElse::make(producedVals, ptrInc);
        }
        util::append(caseBody, {ptrInc});
      }

      indexVars.pop_back();
      cases.push_back({caseExpr, Block::make(caseBody)});
    }
    
    Stmt caseStmt = noMerge ? cases[0].second : Case::make(cases);
    loopBody.push_back(caseStmt);
    loopBody.push_back(BlankLine::make());

    // Emit code to conditionally increment iterator variables
    if (!noMerge) {
      for (auto& step : lpSteps) {
        storage::Iterator iterator = iterators.getIterator(step);

        Expr iteratorVar = iterator.getIteratorVar();
        Expr tensorIdx = tensorIdxVariables.at(step);

        Stmt inc = VarAssign::make(iteratorVar, Add::make(iteratorVar, 1));

        Stmt maybeInc =
            noMerge ? inc : IfThenElse::make(Eq::make(tensorIdx, idx), inc);

        loopBody.push_back(maybeInc);
      }
    }

    Stmt loop;
    if (noMerge) {
      iassert(lpSimplifiedSteps.size() == 1);
      storage::Iterator iterator = iterators.getIterator(lpSimplifiedSteps[0]);

      loop = For::make(iterator.getIteratorVar(),
                       iterator.begin(), iterator.end(), 1,
                       Block::make(loopBody));
    }
    else {
      // Loop until any index has been exchaused
      Expr untilAnyExhausted;
      for (size_t i=0; i < lpSimplifiedSteps.size(); ++i) {
        storage::Iterator iterator = iterators.getIterator(lpSteps[i]);
        Expr indexExhausted =
            Lt::make(iterator.getIteratorVar(), iterator.end());

        untilAnyExhausted = (i == 0)
                            ? indexExhausted
                            : ir::And::make(untilAnyExhausted, indexExhausted);
      }

      loop = While::make(untilAnyExhausted, Block::make(loopBody));
    }
    mergeLoops.push_back(loop);

    if (i < latticePoints.size()-1) {
      mergeLoops.push_back(BlankLine::make());
    }
  }
  if (noMerge) {
    iassert(mergeLoops.size() > 0);
    mergeLoops = {mergeLoops[0]};
  }
  util::append(mergeCode, mergeLoops);

  // Emit code to store the segment size to ptr
  if (util::contains(properties, Assemble) && resultIterator.defined()) {
    Stmt ptrStore = resultIterator.storePtr();
    if (ptrStore.defined()) {
      util::append(mergeCode, {BlankLine::make()});
      if (util::contains(properties, Comment)) {
        util::append(mergeCode, {Comment::make("set "+toString(resultTensorVar)+
                                                ".L"+to_string(layer)+".ptr")});
      }
      util::append(mergeCode, {ptrStore});
    }
  }

  return mergeCode;
}

/// Lower one level of the iteration schedule. Dispatches to specialized lower
/// functions that recursively call this function to lower the next level
/// inside each loop at this level.
vector<Stmt> lower(const Expr& expr,
                   const set<Property>& properties,
                   const IterationSchedule& schedule,
                   const Iterators& iterators,
                   const taco::Expr& indexExpr,
                   size_t layer,
                   vector<Expr> indexVars,
                   map<Tensor,Expr> tensorVars) {
  vector<vector<taco::Var>> layers = schedule.getIndexVariables();
  iassert(layer < layers.size());
  vector<taco::Var> vars = layers[layer];

  vector<Stmt> levelCode;

  // Compute scalar expressions
  if (vars.size() == 0 && util::contains(properties, Compute)) {
    Expr resultTensorVar = tensorVars.at(schedule.getTensor());
    Expr resultPtr = 0;
    Expr vals = GetProperty::make(resultTensorVar, TensorProperty::Values);
    Stmt compute = Store::make(vals, resultPtr, expr);
    util::append(levelCode, {compute});
  }

  // Emit a loop sequence to merge the iteration space of incoming paths, and
  // recurse on the next layer in each loop.
  for (taco::Var var : vars) {
    vector<pair<Expr, Expr>> availableSubExprs;
    auto irExpr = lowerScalarExpression(indexExpr, iterators, schedule,
                                        tensorVars);
    auto unavail = extractAvailableExpressions(irExpr, var, iterators, schedule,
                                               &availableSubExprs);

    vector<Stmt> computeAvailStmts;
    for (auto& avail : availableSubExprs) {
      Stmt computeAvail = VarAssign::make(avail.first, avail.second);
      computeAvailStmts.push_back(computeAvail);
    }

    vector<Stmt> loweredCode = merge(expr, indexExpr, layer, var, indexVars,
                                     properties, schedule, iterators,
                                     tensorVars);
    util::append(levelCode, loweredCode);
  }

  return levelCode;
}

Stmt lower(const Tensor& tensor,
           string funcName, const set<Property>& properties) {
  auto name = tensor.getName();
  auto vars = tensor.getIndexVars();
  auto indexExpr = tensor.getExpr();

  string exprString = name + "(" + util::join(vars) + ")" +
                      " = " + util::toString(indexExpr);

  // Pack the tensor and it's expression operands into the parameter list
  vector<Expr> parameters;
  vector<Expr> results;
  map<Tensor,Expr> tensorVars;

  // Pack result tensor into output parameter list
  Expr tensorVar = Var::make(name, typeOf<double>(), tensor.getFormat());
  tensorVars.insert({tensor, tensorVar});
  parameters.push_back(tensorVar);

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
    MergeRule mergeRule = schedule.getMergeRule(indexVar);

    TensorPathStep step = mergeRule.getResultStep();
    Tensor result = schedule.getResultTensorPath().getTensor();

    Expr tensorVar = tensorVars.at(result);
    storage::Iterator iterator = iterators.getIterator(step);
    Expr ptr = iterator.getPtrVar();
    Expr ptrPrev = iterators.getPreviousIterator(step).getPtrVar();

    // Emit code to initialize the result ptr variable
    Stmt iteratorInit = VarAssign::make(iterator.getPtrVar(), iterator.begin());
    resultPtrInit.push_back(iteratorInit);
  }

  // Lower the scalar expression
  Expr expr = lowerScalarExpression(indexExpr, iterators, schedule, tensorVars);

  // Lower the iteration schedule
  auto loweredCode = lower(expr, properties, schedule, iterators,
                           indexExpr, 0, {}, tensorVars);

  // Create function
  vector<Stmt> body;
  body.push_back(Comment::make(exprString));
  body.insert(body.end(), resultPtrInit.begin(), resultPtrInit.end());
  body.insert(body.end(), loweredCode.begin(), loweredCode.end());

  return Function::make(funcName, parameters, results, Block::make(body));
}

}}
