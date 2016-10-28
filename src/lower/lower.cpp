#include "lower/lower.h"

#include <vector>

#include "lower/tensor_path.h"
#include "lower/merge_rule.h"
#include "lower/merge_lattice.h"
#include "lower/iteration_schedule.h"
#include "lower/iterators.h"

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

vector<Stmt> lower(const set<Property>& properties,
                   const IterationSchedule& schedule,
                   const Iterators& iterators,
                   size_t level,
                   vector<Expr> indexVars,
                   map<Tensor,Expr> tensorVars);

/// Emit code to print the visited index variable coordinates
static vector<Stmt> printCoordinate(const Iterators& iterators,
                                    const vector<Expr>& indexVars) {
  vector<string> fmtstrings(indexVars.size(), "%d");
  string format = util::join(fmtstrings, ",");
  vector<Expr> printvars = indexVars;
  return {Print::make("("+format+")\\n", printvars)};
}

static vector<Stmt> assembleCode(const IterationSchedule &schedule,
                                 const Iterators& iterators,
                                 const vector<Expr>& indexVars) {
  Tensor tensor   = schedule.getTensor();
  taco::Expr expr = tensor.getExpr();

  return {};
}

static vector<Stmt> evaluateCode(const IterationSchedule &schedule,
                                 const Iterators& iterators,
                                 const vector<Expr>& indexVars) {
  return {};
}

static string ptrName(taco::Var var, Tensor tensor) {
  return var.getName() + tensor.getName() + "_ptr";
}

/// Lower a tensor index variable whose values come from a single iteration
/// space. It therefore does not need to merge several tensor paths.
static vector<Stmt> lowerUnmerged(const set<Property>& properties,
                                  taco::Var var,
                                  size_t level,
                                  TensorPathStep step,
                                  const IterationSchedule& schedule,
                                  const Iterators& iterators,
                                  vector<Expr> idxVars,
                                  map<Tensor,Expr> tensorVars) {

  storage::Iterator parentIt = iterators.getParentIterator(step);
  Expr ptrParent = parentIt.getIteratorVar();

  iassert(ptrParent.defined());

  Tensor tensor = step.getPath().getTensor();
  Expr   tvar   = tensorVars.at(tensor);

  // Get the format level of this index variable
  Level levelFormat = tensor.getFormat().getLevels()[step.getStep()];
  int   dim         = levelFormat.getDimension();

  Expr ptr = Var::make(ptrName(var, tensor), typeOf<int>(), false);
  Expr idx = Var::make(var.getName(), typeOf<int>(), false);

  vector<Stmt> loweredCode;
  switch (levelFormat.getType()) {
    case LevelType::Dense: {
      Expr ptrUnpack = GetProperty::make(tvar, TensorProperty::Pointer, dim);
      Expr initVal = ir::Add::make(ir::Mul::make(ptrParent, ptrUnpack), idx);
      Stmt init  = VarAssign::make(ptr, initVal);

      idxVars.push_back(idx);
      auto body = lower(properties, schedule, iterators,
                        level+1, idxVars, tensorVars);

      vector<Stmt> loopBody;
      loopBody.push_back(init);
      loopBody.insert(loopBody.end(), body.begin(), body.end());

      loweredCode = {For::make(idx, 0, ptrUnpack, 1, Block::make(loopBody))};
      break;
    }
    case LevelType::Sparse: {
      Expr ptrUnpack = GetProperty::make(tvar, TensorProperty::Pointer, dim);
      Expr idxUnpack = GetProperty::make(tvar, TensorProperty::Index, dim);
      Expr initVal = Load::make(idxUnpack, ptr);
      Stmt init  = VarAssign::make(idx, initVal);
      Expr loopBegin = Load::make(ptrUnpack, ptrParent);
      Expr loopEnd = Load::make(ptrUnpack, ir::Add::make(ptrParent, 1));

      idxVars.push_back(idx);
      auto body = lower(properties, schedule, iterators,
                        level+1, idxVars, tensorVars);

      vector<Stmt> loopBody;
      loopBody.push_back(init);
      loopBody.insert(loopBody.end(), body.begin(), body.end());

      loweredCode = {For::make(ptr, loopBegin, loopEnd, 1,
                               Block::make(loopBody))};
      break;
    }
    case LevelType::Fixed:
      not_supported_yet;
      break;
  }
  iassert(loweredCode.size() > 0);
  return loweredCode;
}

Stmt initPtr(Expr ptr, Expr parentPtr, Level levelFormat, Expr tensor) {
  Stmt initPtrStmt;
  switch (levelFormat.getType()) {
    case LevelType::Dense: {
      // Merging with dense formats should have been optimized away.
      ierror << "Doesn't make any sense to merge with a dense format.";
      break;
    }
    case LevelType::Sparse: {
      int dim = levelFormat.getDimension();
      Expr ptrArray = GetProperty::make(tensor, TensorProperty::Pointer, dim);
      Expr ptrVal = Load::make(ptrArray, parentPtr);
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

Expr exhausted(Expr ptr, Expr parentPtr, Level levelFormat, Expr tensor) {
  Expr exhaustedExpr;
  switch (levelFormat.getType()) {
    case LevelType::Dense: {
      // Merging with dense formats should have been optimized away.
      ierror << "Doesn't make any sense to merge with a dense format.";
      break;
    }
    case LevelType::Sparse: {
      int dim = levelFormat.getDimension();
      Expr ptrArray = GetProperty::make(tensor, TensorProperty::Pointer, dim);
      Expr ptrVal = Load::make(ptrArray, ir::Add::make(parentPtr,1));
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
      int dim = levelFormat.getDimension();
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
  Stmt incStmt = VarAssign::make(ptr, ir::Add::make(ptr,1));
  return IfThenElse::make(test, incStmt);
}

static vector<Stmt> lowerMerged(size_t level,
                                taco::Var var,
                                const map<TensorPathStep,Expr>& parentPtrs,
                                vector<Expr> indexVars,
                                MergeRule mergeRule,
                                const set<Property>& properties,
                                const IterationSchedule& schedule,
                                const Iterators& iterators,
                                const map<Tensor,Expr>& tensorVars) {

  auto steps = mergeRule.getSteps();
  auto mergeLattice = buildMergeLattice(mergeRule);

  std::cout << std::endl << "# Lattice" << std::endl;
  std::cout << mergeLattice << std::endl;

  vector<Stmt> mergeLoops;

  // Initialize ptr variables
  map<TensorPathStep, Expr> tensorPtrVariables;
  for (auto& parentPtrPair : parentPtrs) {
    TensorPathStep step = parentPtrPair.first;
    Tensor tensor = step.getPath().getTensor();

    Expr ptr = Var::make(ptrName(var, tensor), typeOf<int>(), false);
    tensorPtrVariables.insert({step, ptr});

    Expr parentPtr = parentPtrPair.second;
    Level levelFormat = tensor.getFormat().getLevels()[step.getStep()];
    Expr tvar = tensorVars.at(tensor);

    Stmt initPtrStmt = initPtr(ptr, parentPtr, levelFormat, tvar);
    mergeLoops.push_back(initPtrStmt);
  }
  
  // Emit one loop per lattice point lp
  auto latticePoints = mergeLattice.getPoints();
  for (size_t i=0; i < latticePoints.size(); ++i) {
    auto lp = latticePoints[i];

    vector<Stmt> loopBody;
    auto steps = lp.getSteps();

    // Iterate until any index has been exchaused
    Expr untilAnyExhausted;
    for (size_t i=0; i < steps.size(); ++i) {
      auto step = steps[i];
      Tensor tensor = step.getPath().getTensor();
      Expr ptr = tensorPtrVariables.at(step);
      Expr parentPtr = parentPtrs.at(step);
      Level levelFormat = tensor.getFormat().getLevels()[step.getStep()];
      Expr tvar = tensorVars.at(tensor);

      Expr indexExhausted = exhausted(ptr, parentPtr, levelFormat, tvar);
      untilAnyExhausted = (i == 0)
                          ? indexExhausted
                          : ir::And::make(untilAnyExhausted, indexExhausted);
    }

    // Emit code to initialize path index variables
    map<TensorPathStep, Expr> tensorIdxVariables;
    vector<Expr> tensorIdxVariablesVector;
    for (auto& step : steps) {
      Expr ptr = tensorPtrVariables.at(step);
      Tensor tensor = step.getPath().getTensor();
      Expr tvar = tensorVars.at(tensor);

      Expr tensorIdx = Var::make(var.getName()+tensor.getName(),
                                 typeOf<int>(), false);
      tensorIdxVariables.insert({step, tensorIdx});
      tensorIdxVariablesVector.push_back(tensorIdx);

      Stmt initTensorIndexStmt = initTensorIdx(tensorIdx, ptr, tvar, step);
      loopBody.push_back(initTensorIndexStmt);
    }

    // Emit code to initialize the index variable (min of path index variables)
    Expr idx = Var::make(var.getName(), typeOf<int>(), false);
    Stmt initIdxStmt = initIdx(idx, tensorIdxVariablesVector);
    loopBody.push_back(initIdxStmt);
    loopBody.push_back(BlankLine::make());

    // Emit an elseif per lattice point lq (non-strictly) dominated by lp
    auto dominatedPoints = mergeLattice.getDominatedPoints(lp);
    vector<pair<Expr,Stmt>> cases;
    for (auto& lq : dominatedPoints) {
      auto steps = lq.getSteps();

      Expr caseExpr;
      iassert(steps.size() > 0);
      for (size_t i=0; i < steps.size(); ++i) {
        auto step = steps[i];
        Expr caseTerm = Eq::make(tensorIdxVariables.at(step), idx);
        caseExpr = (i == 0) ? caseTerm : ir::And::make(caseExpr, caseTerm);
      }

      indexVars.push_back(idx);
      auto caseStmts = lower(properties, schedule, iterators, level+1,
                             indexVars, tensorVars);
      indexVars.pop_back();
      cases.push_back({caseExpr, Block::make(caseStmts)});
    }
    Stmt casesStmt = Case::make(cases);
    loopBody.push_back(casesStmt);
    loopBody.push_back(BlankLine::make());

    // Emit code to conditionally increment ptr variables
    for (auto& step : steps) {
      Expr ptr = tensorPtrVariables.at(step);
      Expr tensorIdx = tensorIdxVariables.at(step);

      Stmt advanceStmt = advance(tensorIdx, idx, ptr);
      loopBody.push_back(advanceStmt);
    }

    mergeLoops.push_back(While::make(untilAnyExhausted, Block::make(loopBody)));

    if (i < latticePoints.size()-1) {
      mergeLoops.push_back(BlankLine::make());
    }
  }

  return mergeLoops;
}

/// Lower one level of the iteration schedule. Dispatches to specialized lower
/// functions that recursively call this function to lower the next level
/// inside each loop at this level.
vector<Stmt> lower(const set<Property>& properties,
                   const IterationSchedule& schedule,
                   const Iterators& iterators,
                   size_t level,
                   vector<Expr> idxVars,
                   map<Tensor,Expr> tensorVars) {
  vector<vector<taco::Var>> levels = schedule.getIndexVariables();

  vector<Stmt> levelCode;

  // Base case: emit code to assemble, evaluate or debug print the tensor.
  if (level == levels.size()) {
    if (util::contains(properties, Print)) {
      auto print = printCoordinate(iterators, idxVars);
      levelCode.insert(levelCode.end(), print.begin(), print.end());
    }

    if (util::contains(properties, Assemble)) {
      auto assemble = assembleCode(schedule, iterators, idxVars);
      levelCode.insert(levelCode.end(), assemble.begin(), assemble.end());
    }

    if (util::contains(properties, Evaluate)) {
      auto evaluate = evaluateCode(schedule, iterators, idxVars);
      levelCode.insert(levelCode.end(), evaluate.begin(), evaluate.end());
    }

    return levelCode;
  }

  // Recursive case: emit a loop sequence to merge the iteration space of
  //                 incoming paths, and recurse on the next level in each loop.
  iassert(level < levels.size());

  vector<taco::Var> vars  = levels[level];
  for (taco::Var var : vars) {
    vector<Stmt> varCode;

    MergeRule mergeRule = schedule.getMergeRule(var);
    vector<TensorPathStep> steps = mergeRule.getSteps();

    // If there's only one incoming path then we emit a for loop.
    // Otherwise, we emit while loops that merge the incoming paths.
    if (steps.size() == 1) {
      vector<Stmt> loweredCode = lowerUnmerged(properties,
                                               var,
                                               level,
                                               steps[0],
                                               schedule,
                                               iterators,
                                               idxVars,
                                               tensorVars);
      varCode.insert(varCode.end(), loweredCode.begin(), loweredCode.end());
    }
    else {
      map<TensorPathStep, Expr> parentPtrs;
      for (auto& step : steps) {
        parentPtrs.insert({step, 0});
      }

      vector<Stmt> loweredCode = lowerMerged(level,
                                             var,
                                             parentPtrs,
                                             idxVars,
                                             mergeRule,
                                             properties,
                                             schedule,
                                             iterators,
                                             tensorVars);
      varCode.insert(varCode.end(), loweredCode.begin(), loweredCode.end());
    }
    levelCode.insert(levelCode.end(), varCode.begin(), varCode.end());
  }

  return levelCode;
}

static inline tuple<vector<Expr>, vector<Expr>, map<Tensor,Expr>>
createParameters(const Tensor& tensor) {

  vector<Tensor> operands = internal::getOperands(tensor.getExpr());
  map<Tensor,Expr> tensorVariables;

  // Build parameter list
  vector<Expr> parameters;
  for (auto& operand : operands) {
    iassert(!util::contains(tensorVariables, operand));

    Expr operandVar = Var::make(operand.getName(), typeOf<double>(),
                                operand.getFormat());
    tensorVariables.insert({operand, operandVar});
    parameters.push_back(operandVar);
  }

  // Build results parameter list
  vector<Expr> results;
  Expr tensorVar = Var::make(tensor.getName(), typeOf<double>(),
                             tensor.getFormat());
  tensorVariables.insert({tensor, tensorVar});
  results.push_back(tensorVar);

  return tuple<vector<Expr>, vector<Expr>, map<Tensor,Expr>>
      {parameters, results, tensorVariables};
}

Stmt lower(const Tensor& tensor,
           const set<Property>& properties,
           string funcName) {
  string exprString = tensor.getName()
                    + "(" + util::join(tensor.getIndexVars()) + ")"
                    + " = " + util::toString(tensor.getExpr());

  IterationSchedule schedule = IterationSchedule::make(tensor);

  vector<Expr> parameters;
  vector<Expr> results;
  map<Tensor,Expr> tensorVariables;
  tie(parameters, results, tensorVariables) = createParameters(tensor);

  Iterators iterators(schedule, tensorVariables);

  // Lower the iteration schedule
  vector<Stmt> loweredCode = lower(properties, schedule, iterators,
                                   0, {}, tensorVariables);

  // Create function
  vector<Stmt> body;
  body.push_back(Comment::make(exprString));
  body.insert(body.end(), loweredCode.begin(), loweredCode.end());

  return Function::make(funcName, parameters, results, Block::make(body));
}

}}
