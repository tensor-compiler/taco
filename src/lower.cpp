#include "lower.h"

#include "internal_tensor.h"
#include "expr.h"
#include "operator.h"
#include "component_types.h"
#include "ir.h"
#include "var.h"
#include "iteration_schedule/iteration_schedule.h"
#include "iteration_schedule/merge_rule.h"
#include "util/strings.h"

using namespace std;

namespace taco {
namespace internal {

Stmt lower(const is::IterationSchedule& schedule, size_t level) {
  vector<Stmt> levelCode;
  iassert(level < schedule.getIndexVariables().size());

  vector<taco::Var> vars  = schedule.getIndexVariables()[level];
  for (taco::Var var : vars) {
    vector<Stmt> varCode;

    // For each var in the iteration schedule level we emit code to produce it's
    // values. The emitted code must merge all incomming paths according to the
    // var's merge rule.

    is::MergeRule mergeRule = schedule.getMergeRule(var);
    std::cout << mergeRule << ":" << std::endl;

    Expr pathIndexVar = Var::make(var.getName(), typeOf<int>(), false);
    Expr segmentVar   = Var::make(var.getName()+var.getName(), typeOf<int>(),
                                  false);

    Stmt begin = VarAssign::make(pathIndexVar, 0);
    Expr end   = Lte::make(pathIndexVar, 10);
    Stmt inc   = VarAssign::make(pathIndexVar, Add::make(pathIndexVar, 1));
    Stmt init  = VarAssign::make(segmentVar, pathIndexVar);

    vector<Stmt> loopBody;
    loopBody.push_back(init);
    loopBody.push_back(inc);
    Stmt loop = While::make(end, Block::make(loopBody));


    levelCode.push_back(begin);
    levelCode.push_back(loop);
    levelCode.insert(levelCode.begin(), varCode.begin(), varCode.end());
  }

  return Block::make(levelCode);
}

Stmt lower(const internal::Tensor& tensor, LowerKind lowerKind) {
  auto expr     = tensor.getExpr();
  auto schedule = is::IterationSchedule::make(tensor);

  // Lower the iteration schedule
  Stmt body = lower(schedule, 0);
  std::cout << body << std::endl;

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
  auto var = Var::make("x", typeOf<int>());
  auto func = Function::make(funcName,
                             {Var::make("y", typeOf<double>())}, {var},
                             Block::make({Store::make(var, Literal::make(0),
                                                      Literal::make(99))}));
  return func;
}

}}
