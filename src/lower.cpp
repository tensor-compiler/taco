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
#include "iteration_schedule/iteration_schedule.h"
#include "util/collections.h"
#include "util/strings.h"

using namespace std;

namespace taco {
namespace internal {

struct TensorVariables {
  vector<Expr> dimensions;
};

vector<Stmt> lower(const is::IterationSchedule& schedule, size_t level,
                   Expr parentSegmentVar, vector<Expr> indexVars,
                   map<Tensor,TensorVariables> tensorVars);

/// Emit code to print the index variables (coordinate) and innermost ptr.
static vector<Stmt> emitPrint(Expr segmentVar, const vector<Expr>& indexVars) {
  vector<string> fmtstrings(indexVars.size(), "%d");
  string format = util::join(fmtstrings, ",");
  vector<Expr> printvars = indexVars;
  printvars.push_back(segmentVar);
  return {Print::make("("+format+"): %d\\n", printvars)};
}

/// Lower a tensor index variable whose values come from a single iteration
/// space. It therefore does not need to merge several tensor paths.
static vector<Stmt> lowerUnmerged(taco::Var var,
                                  size_t level,
                                  is::TensorPath path,
                                  const is::IterationSchedule& schedule,
                                  Expr ptrParent,
                                  vector<Expr> idxVars,
                                  map<Tensor,TensorVariables> tensorVars) {
  auto tensor = path.getTensor();
  auto tvars  = tensorVars.at(tensor);
  auto dim    = tvars.dimensions[level];

  // Get the format level of this index variable
  size_t loc = 0;
  auto pathVars = path.getPath();
  for (size_t i=0; i < pathVars.size(); ++i) {
    auto pathVar = pathVars[i];
    if (pathVar == var) {
      loc = i;
      break;
    }
  }
  auto formatLevel = tensor.getFormat().getLevels()[loc];
  std::cout << formatLevel << std::endl;

  vector<Stmt> loweredCode;
  switch (formatLevel.getType()) {
    case LevelType::Dense: {
      Expr ptr = Var::make(var.getName()+"ptr", typeOf<int>(), false);
      Expr idx = Var::make(var.getName(), typeOf<int>(), false);

      Expr initVal = (ptrParent.defined())
                   ? Add::make(Mul::make(ptrParent, dim), idx)
                   : idx;
      Stmt init = VarAssign::make(ptr, initVal);

      idxVars.push_back(idx);
      auto body = lower(schedule, level+1, ptr, idxVars, tensorVars);

      vector<Stmt> loopBody;
      loopBody.push_back(init);
      loopBody.insert(loopBody.end(), body.begin(), body.end());

      loweredCode = {For::make(idx, 0, dim, 1, Block::make(loopBody))};
      break;
    }
    case LevelType::Sparse:
      not_supported_yet;
      break;
    case LevelType::Fixed:
      not_supported_yet;
      break;
  }
  iassert(loweredCode.size() > 0);
  return loweredCode;
}

static vector<Stmt> lowerMerged() {
//      is::TensorPath path = getIncomingPaths.paths[0];
//
//      TensorVariables tvars = tensorVars.at(path.getTensor());
//      if (level == 0) {
//        vector<string> fmtstrings(tvars.dimensions.size(), "%d");
//        string format = util::join(fmtstrings, "x");
//        varCode.push_back(Print::make(format + "\\n", tvars.dimensions));
//      }
//      auto dim = tvars.dimensions[level];
//
//      Expr segmentVar   = Var::make(var.getName()+var.getName(), typeOf<int>(),
//                                    false);
//      Expr pathIndexVar = Var::make(var.getName(), typeOf<int>(), false);
//      Expr indexVar = pathIndexVar;
//      indexVars.push_back(indexVar);
//
//      Stmt begin = VarAssign::make(pathIndexVar, 0);
//      Expr end   = Lt::make(pathIndexVar, dim);
//      Stmt inc   = VarAssign::make(pathIndexVar, Add::make(pathIndexVar, 1));
//
//      Expr initVal = (parentSegmentVar.defined())
//                   ? Add::make(Mul::make(parentSegmentVar, dim), pathIndexVar)
//                   : pathIndexVar;
//      Stmt init = VarAssign::make(segmentVar, initVal);
//
//      vector<Stmt> loopBody;
//      loopBody.push_back(init);
//      if (level < (levels.size()-1)) {
//        vector<Stmt> body = lower(schedule, level+1, segmentVar, indexVars,
//                                  tensorVars);
//        loopBody.insert(loopBody.end(), body.begin(), body.end());
//      }
//      else {
//        vector<string> fmtstrings(indexVars.size(), "%d");
//        string format = util::join(fmtstrings, ",");
//        vector<Expr> printvars = indexVars;
//        printvars.push_back(segmentVar);
//        Stmt print = Print::make("("+format+"): %d\\n", printvars);
//        loopBody.push_back(print);
//      }
//
//      loopBody.push_back(inc);
//      Stmt loop = While::make(end, Block::make(loopBody));
//
//      varCode.push_back(begin);
//      varCode.push_back(loop);
//      levelCode.insert(levelCode.end(), varCode.begin(), varCode.end());
  return {};
}

/// Lower one level of the iteration schedule. Dispatches to specialized lower
/// functions that recursively call this function to lower the next level
/// inside each loop at this level.
vector<Stmt> lower(const is::IterationSchedule& schedule, size_t level,
                   Expr ptrParent, vector<Expr> idxVars,
                   map<Tensor,TensorVariables> tensorVars) {
  vector<vector<taco::Var>> levels = schedule.getIndexVariables();

  vector<Stmt> levelCode;

  // Base case: emit code to assemble, evaluate or debug print the tensor.
  if (level == levels.size()) {
    return emitPrint(ptrParent, idxVars);
    // return emitAssemble();
    // return emitEvaluate();
  }

  // Recursive case: emit a loop sequence to merge the iteration space of
  //                 incoming paths, and recurse on the next level in each loop.
  iassert(level < levels.size());

  vector<taco::Var> vars  = levels[level];
  for (taco::Var var : vars) {
    vector<Stmt> varCode;

    is::MergeRule mergeRule = schedule.getMergeRule(var);

    struct GetMergedPaths : public is::MergeRuleVisitor {
      vector<is::TensorPath> paths;
      void visit(const is::Path* rule) {
        paths.push_back(rule->path);
      }
    };
    GetMergedPaths getIncomingPaths;
    mergeRule.accept(&getIncomingPaths);

    // If there's only one incoming path then we emit a for loop.
    // Otherwise, we emit while loops that merge the incoming paths.
    if (getIncomingPaths.paths.size() == 1) {
      vector<Stmt> loweredCode = lowerUnmerged(var, level,
                                               getIncomingPaths.paths[0],
                                               schedule,
                                               ptrParent,
                                               idxVars, tensorVars);
      varCode.insert(varCode.end(), loweredCode.begin(), loweredCode.end());
    }
    else {
      terror << "merging not supported";
      vector<Stmt> loweredCode = lowerMerged();
      varCode.insert(varCode.end(), loweredCode.begin(), loweredCode.end());
    }
    levelCode.insert(levelCode.end(), varCode.begin(), varCode.end());
  }

  return levelCode;
}

Stmt lower(const internal::Tensor& tensor, LowerKind lowerKind) {
  string exprString = tensor.getName()
                    + "(" + util::join(tensor.getIndexVars()) + ")"
                    + " = " + util::toString(tensor.getExpr());

  auto expr     = tensor.getExpr();
  auto schedule = is::IterationSchedule::make(tensor);

  vector<Tensor> operands = getOperands(tensor.getExpr());

  map<Tensor,TensorVariables> tensorVariables;
  for (auto& operand : operands) {
    iassert(!util::contains(tensorVariables, operand));
    TensorVariables tvars;
    for (size_t i=0; i < operand.getOrder(); ++i) {
      Expr dimi = Var::make(tensor.getName() + "_d" + to_string(i),
                            typeOf<int>(), false);
      tvars.dimensions.push_back(dimi);
    }
    tensorVariables.insert({operand, tvars});
  }

  // Lower the iteration schedule
  vector<Stmt> loweredCode = lower(schedule, 0, Expr(), {}, tensorVariables);

  // Determine the function name
  string funcName;
  switch (lowerKind) {
    case LowerKind::Assemble:
      funcName = "assemble";
      break;
    case LowerKind::Evaluate:
      funcName = "evaluate";
      break;
    case LowerKind::AssembleAndEvaluate:
      funcName = "assemble_evaluate";
      break;
    case LowerKind::Print:
      funcName = "print";
      break;
  }
  iassert(funcName != "");

  // Build argument list
  vector<Expr> arguments;
  for (auto& operand : operands) {
    TensorVariables tvars = tensorVariables.at(operand);

    // Insert operand dimensions
    arguments.insert(arguments.end(),
                     tvars.dimensions.begin(), tvars.dimensions.end());
  }

  // Build result list
  vector<Expr> results;

  // Create function
  vector<Stmt> body;
  body.push_back(Comment::make(exprString));
  body.insert(body.end(), loweredCode.begin(), loweredCode.end());

  return Function::make(funcName, arguments, results, Block::make(body));
}

}}
