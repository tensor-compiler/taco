#include "lower.h"

#include <vector>

#include "internal_tensor.h"
#include "expr.h"
#include "operator.h"
#include "component_types.h"
#include "ir.h"
#include "var.h"
#include "iteration_schedule/tensor_path.h"
#include "iteration_schedule/merge_rule.h"
#include "iteration_schedule/merge_lattice.h"
#include "iteration_schedule/iteration_schedule.h"
#include "util/collections.h"
#include "util/strings.h"

using namespace std;

namespace taco {
namespace internal {

using namespace taco::ir;
using taco::ir::Expr;
using taco::ir::Var;

vector<Stmt> lower(const set<Property>& properties,
                   const is::IterationSchedule& schedule,
                   size_t level,
                   Expr parentPtr,
                   vector<Expr> indexVars,
                   map<Tensor,Expr> tensorVars);

/// Emit code to print the visited index variable coordinates
static vector<Stmt> printCode(const vector<Expr>& indexVars, Expr ptr) {
  vector<string> fmtstrings(indexVars.size(), "%d");
  string format = util::join(fmtstrings, ",");
  vector<Expr> printvars = indexVars;
  printvars.push_back(ptr);
  return {Print::make("("+format+"): %d\\n", printvars)};
}

static vector<Stmt> assembleCode(const is::IterationSchedule &schedule,
                                 const vector<Expr>& indexVars, Expr ptr) {
  Tensor tensor   = schedule.getTensor();
  taco::Expr expr = tensor.getExpr();

  return {};
}

static vector<Stmt> evaluateCode(const is::IterationSchedule &schedule,
                                 const vector<Expr>& indexVars, Expr ptr) {
  return {};
}

/// Lower a tensor index variable whose values come from a single iteration
/// space. It therefore does not need to merge several tensor paths.
static vector<Stmt> lowerUnmerged(const set<Property>& properties,
                                  taco::Var var,
                                  size_t level,
                                  is::TensorPath path,
                                  const is::IterationSchedule& schedule,
                                  Expr ptrParent,
                                  vector<Expr> idxVars,
                                  map<Tensor,Expr> tensorVars) {
  iassert(ptrParent.defined());

  auto tensor = path.getTensor();
  auto tvar  = tensorVars.at(tensor);

  // Get the format level of this index variable
  size_t loc = 0;
  auto pathVars = path.getVariables();
  for (size_t i=0; i < pathVars.size(); ++i) {
    auto pathVar = pathVars[i];
    if (pathVar == var) {
      loc = i;
      break;
    }
  }
  auto formatLevel = tensor.getFormat().getLevels()[loc];
  int dim = formatLevel.getDimension();

  Expr ptr = Var::make(var.getName()+"ptr", typeOf<int>(), false);
  Expr idx = Var::make(var.getName(), typeOf<int>(), false);

  vector<Stmt> loweredCode;
  switch (formatLevel.getType()) {
    case LevelType::Dense: {
      Expr ptrUnpack = GetProperty::make(tvar, TensorProperty::Pointer, dim);
      Expr initVal = ir::Add::make(ir::Mul::make(ptrParent, ptrUnpack), idx);
      Stmt init  = VarAssign::make(ptr, initVal);

      idxVars.push_back(idx);
      auto body = lower(properties, schedule, level+1, ptr, idxVars,
                        tensorVars);

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
      auto body = lower(properties, schedule, level+1, ptr, idxVars,
                        tensorVars);

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

static vector<Stmt> lowerMerged(size_t level,
                                taco::Var var,
                                const map<is::TensorPath,Expr>& parentPtrs,
                                vector<Expr> idxVars,
                                is::MergeRule mergeRule,
                                const set<Property>& properties,
                                const is::IterationSchedule& schedule,
                                const map<Tensor,Expr>& tensorVars) {

  is::MergeLattice mergeLattice = buildMergeLattice(mergeRule);

  std::cout << std::endl << "# Lattice" << std::endl;
  std::cout << mergeLattice << std::endl;

  vector<Stmt> mergeLoops;

  // Initialize ptr variables
  // ...
  
  // Emit one loop per lattice point lp
  for (auto& lp : mergeLattice.getPoints()) {
    // Initialize path index variables
    // ...

    // Initialize the index variable (min of path index variables)
    // ...

    // Emit an elseif per lattice point lq (non-strictly) dominated by lp
    // ...
  }

  // Conditionally increment ptr variables
  // ...

  return mergeLoops;
}

/// Lower one level of the iteration schedule. Dispatches to specialized lower
/// functions that recursively call this function to lower the next level
/// inside each loop at this level.
vector<Stmt> lower(const set<Property>& properties,
                   const is::IterationSchedule& schedule,
                   size_t level,
                   Expr ptrParent,
                   vector<Expr> idxVars,
                   map<Tensor,Expr> tensorVars) {
  vector<vector<taco::Var>> levels = schedule.getIndexVariables();

  vector<Stmt> levelCode;

  // Base case: emit code to assemble, evaluate or debug print the tensor.
  if (level == levels.size()) {
    if (util::contains(properties, Print)) {
      auto print = printCode(idxVars, ptrParent);
      levelCode.insert(levelCode.end(), print.begin(), print.end());
    }

    if (util::contains(properties, Assemble)) {
      auto assemble = assembleCode(schedule, idxVars, ptrParent);
      levelCode.insert(levelCode.end(), assemble.begin(), assemble.end());
    }

    if (util::contains(properties, Evaluate)) {
      auto evaluate = evaluateCode(schedule, idxVars, ptrParent);
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

    is::MergeRule mergeRule = schedule.getMergeRule(var);
    vector<is::TensorPath> paths = mergeRule.getPaths();

    // If there's only one incoming path then we emit a for loop.
    // Otherwise, we emit while loops that merge the incoming paths.
    if (paths.size() == 1) {
      vector<Stmt> loweredCode = lowerUnmerged(properties,
                                               var,
                                               level,
                                               paths[0],
                                               schedule,
                                               ptrParent,
                                               idxVars,
                                               tensorVars);
      varCode.insert(varCode.end(), loweredCode.begin(), loweredCode.end());
    }
    else {
      map<is::TensorPath, Expr> parentPtrs;
      for (auto& path : paths) {
        parentPtrs.insert({path, 0});
      }

      vector<Stmt> loweredCode = lowerMerged(level,
                                             var,
                                             parentPtrs,
                                             idxVars,
                                             mergeRule,
                                             properties,
                                             schedule,
                                             tensorVars);
      varCode.insert(varCode.end(), loweredCode.begin(), loweredCode.end());
    }
    levelCode.insert(levelCode.end(), varCode.begin(), varCode.end());
  }

  return levelCode;
}

static inline tuple<vector<Expr>, vector<Expr>, map<Tensor,Expr>>
createParameters(const Tensor& tensor) {

  vector<Tensor> operands = getOperands(tensor.getExpr());
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

  auto schedule = is::IterationSchedule::make(tensor);

  vector<Expr> parameters;
  vector<Expr> results;
  map<Tensor,Expr> tensorVariables;
  tie(parameters, results, tensorVariables) = createParameters(tensor);

  // Lower the iteration schedule
  vector<Stmt> loweredCode = lower(properties, schedule,
                                   0, Expr(0), {}, tensorVariables);

  // Create function
  vector<Stmt> body;
  body.push_back(Comment::make(exprString));
  body.insert(body.end(), loweredCode.begin(), loweredCode.end());

  return Function::make(funcName, parameters, results, Block::make(body));
}

}}
