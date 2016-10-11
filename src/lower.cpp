#include "lower.h"

#include "internal_tensor.h"
#include "expr.h"
#include "operator.h"
#include "iteration_schedule.h"
#include "ir.h"
#include "util/strings.h"

using namespace std;

namespace taco {
namespace internal {

Stmt lower(const internal::Tensor& tensor, LowerKind lowerKind) {
  taco::Expr expr = tensor.getExpr();
  auto schedule = IterationSchedule::make(expr);

  Stmt body = Block::make();

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
