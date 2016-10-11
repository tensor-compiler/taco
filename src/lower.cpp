#include "lower.h"

#include "internal_tensor.h"
#include "expr.h"
#include "operator.h"
#include "iteration_schedule.h"
#include "component_types.h"
#include "ir.h"
#include "util/strings.h"

using namespace std;

namespace taco {
namespace internal {

Stmt lower(const IterationSchedule& schedule, size_t level) {
  vector<Stmt> code;
  iassert(level < schedule.getIndexVariables().size());

  vector<taco::Var> vars  = schedule.getIndexVariables()[level];
  for (taco::Var var : vars) {
    Expr i = Var::make(var.getName(), typeOf<int>(), false);
    Stmt loop = For::make(i, 0, 10, 1, Block::make());

    code.push_back(loop);
  }

  return Block::make(code);
}

Stmt lower(const internal::Tensor& tensor, LowerKind lowerKind) {
  auto expr     = tensor.getExpr();
  auto schedule = IterationSchedule::make(expr);

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
