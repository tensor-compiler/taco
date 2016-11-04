#include "lower_scalar_expression.h"

#include "iterators.h"
#include "iteration_schedule.h"
#include "tensor_path.h"

#include "internal_tensor.h"
#include "expr.h"
#include "expr_nodes.h"
#include "expr_visitor.h"

using namespace std;
using namespace taco::ir;
using taco::internal::Tensor;

namespace taco {
namespace lower {

ir::Expr lowerScalarExpression(const taco::Expr& indexExpr,
                               const Iterators& iterators,
                               const IterationSchedule& schedule,
                               const map<Tensor,ir::Expr>& tensorVars) {

  class LowerVisitor : public internal::ExprVisitorStrict {
  public:
    const Iterators& iterators;
    const IterationSchedule& schedule;
    const map<Tensor,ir::Expr>& tensorVars;
    LowerVisitor(const Iterators& iterators,
                 const IterationSchedule& schedule,
                 const map<Tensor,ir::Expr>& tensorVars)
        : iterators(iterators), schedule(schedule), tensorVars(tensorVars) {
    }

    ir::Expr expr;
    ir::Expr lower(const taco::Expr& indexExpr) {
      indexExpr.accept(this);
      auto e = expr;
      expr = ir::Expr();
      return e;
    }

    void visit(const internal::Read* op) {
      storage::Iterator iterator;
      if (op->tensor.getOrder() == 0) {
        iterator = iterators.getRootIterator();
      }
      else {
        TensorPath path = schedule.getTensorPath(op);
        iterator = iterators.getIterator(path.getLastStep());
      }

      ir::Expr ptr = iterator.getPtrVar();
      ir::Expr tensorVar = tensorVars.at(op->tensor);
      ir::Expr values = GetProperty::make(tensorVar, TensorProperty::Values);

      ir::Expr loadValue = Load::make(values, ptr);
      expr = loadValue;
    }

    void visit(const internal::Neg* op) {
      expr = ir::Neg::make(lower(op->a));
    }

    void visit(const internal::Sqrt* op) {
      not_supported_yet;
    }

    void visit(const internal::Add* op) {
      expr = ir::Add::make(lower(op->a), lower(op->b));
    }

    void visit(const internal::Sub* op) {
      expr = ir::Sub::make(lower(op->a), lower(op->b));
    }

    void visit(const internal::Mul* op) {
      expr = ir::Mul::make(lower(op->a), lower(op->b));
    }

    void visit(const internal::Div* op) {
      expr = ir::Div::make(lower(op->a), lower(op->b));
    }

    void visit(const internal::IntImm* op) {
      expr = ir::Expr(op->val);
    }

    void visit(const internal::FloatImm* op) {
      expr = ir::Expr(op->val);
    }

    void visit(const internal::DoubleImm* op) {
      expr = ir::Expr(op->val);
    }
  };

  return LowerVisitor(iterators, schedule, tensorVars).lower(indexExpr);
}

}}
