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
                   map<Tensor,TensorVariables> tensorVars) {
  vector<Stmt> levelCode;

  iassert(level < schedule.getIndexVariables().size());

  vector<vector<taco::Var>> levels = schedule.getIndexVariables();
  vector<taco::Var> vars  = levels[level];
  for (taco::Var var : vars) {
    vector<Stmt> varCode;

    is::MergeRule mergeRule = schedule.getMergeRule(var);

    struct GetIncomingPaths : public is::MergeRuleVisitor {
      vector<is::TensorPath> paths;
      void visit(const is::Path* rule) {
        paths.push_back(rule->path);
      }
    };
    GetIncomingPaths getIncomingPaths;
    mergeRule.accept(&getIncomingPaths);

    tassert(getIncomingPaths.paths.size() == 1);

    is::TensorPath path = getIncomingPaths.paths[0];

    TensorVariables tvars = tensorVars.at(path.getTensor());
    if (level == 0) {
      vector<string> fmtstrings(tvars.dimensions.size(), "%d");
      string format = util::join(fmtstrings, "x");
      varCode.push_back(Print::make(format + "\\n", tvars.dimensions));
    }
    auto dim = tvars.dimensions[level];

    Expr segmentVar   = Var::make(var.getName()+var.getName(), typeOf<int>(),
                                  false);
    Expr pathIndexVar = Var::make(var.getName(), typeOf<int>(), false);
    Expr indexVar = pathIndexVar;
    indexVars.push_back(indexVar);

    Stmt begin = VarAssign::make(pathIndexVar, 0);
    Expr end   = Lt::make(pathIndexVar, dim);
    Stmt inc   = VarAssign::make(pathIndexVar, Add::make(pathIndexVar, 1));

    Expr initVal = (parentSegmentVar.defined())
                   ? Add::make(Mul::make(parentSegmentVar, dim), pathIndexVar)
                   : pathIndexVar;
    Stmt init = VarAssign::make(segmentVar, initVal);

    vector<Stmt> loopBody;
    loopBody.push_back(init);
    if (level < (levels.size()-1)) {
      vector<Stmt> body = lower(schedule, level+1, segmentVar, indexVars,
                                tensorVars);
      loopBody.insert(loopBody.end(), body.begin(), body.end());
    }
    else {
      vector<string> fmtstrings(indexVars.size(), "%d");
      string format = util::join(fmtstrings, ",");
      vector<Expr> printvars = indexVars;
      printvars.push_back(segmentVar);
      Stmt print = Print::make("("+format+"): %d\\n", printvars);
      loopBody.push_back(print);
    }

    loopBody.push_back(inc);
    Stmt loop = While::make(end, Block::make(loopBody));

    varCode.push_back(begin);
    varCode.push_back(loop);
    levelCode.insert(levelCode.end(), varCode.begin(), varCode.end());
  }

  return levelCode;
}

Stmt lower(const internal::Tensor& tensor, LowerKind lowerKind) {
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
  vector<Stmt> body = lower(schedule, 0, Expr(), {}, tensorVariables);

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
  auto func = Function::make(funcName, arguments, results, Block::make(body));
  std::cout << func << std::endl;
  return func;
}

}}
