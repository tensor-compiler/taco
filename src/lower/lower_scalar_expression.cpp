#include "lower_scalar_expression.h"

#include <vector>
#include <set>

#include "iterators.h"
#include "iteration_schedule.h"
#include "tensor_path.h"
#include "merge_rule.h"

#include "internal_tensor.h"
#include "expr.h"
#include "expr_nodes.h"
#include "expr_visitor.h"

#include "ir.h"
#include "ir_visitor.h"

#include "util/collections.h"

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
      expr = ir::Sqrt::make(lower(op->a));
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

/// Extract the sub-expressions that have become available to be computed.
/// These are the sub-expressions where `var` is the variable associated with
/// the last storage dimension of all operands
ir::Expr extractAvailableExpressions(ir::Expr expr, taco::Var var,
                                     const Iterators& iterators,
                                     const IterationSchedule& schedule,
                                     vector<pair<ir::Expr, ir::Expr>>* subExprs) {
  struct ExtractAvailableExpressions : public IRVisitor {
    int i = 0;
    taco::Var var;
    const Iterators& iterators;
    const IterationSchedule& schedule;
    vector<pair<ir::Expr, ir::Expr>>* subExprs;

    set<ir::Expr> currentVars;

    ExtractAvailableExpressions(taco::Var var,
                                const Iterators& iterators,
                                const IterationSchedule& schedule,
                                vector<pair<ir::Expr, ir::Expr>>* subExprs)
        : var(var), iterators(iterators), schedule(schedule),
          subExprs(subExprs) {

      auto steps = schedule.getMergeRule(var).getSteps();
      for (auto& step : steps) {
        auto iterator = iterators.getIterator(step);
        currentVars.insert(iterator.getPtrVar());
      }
    }

    ir::Expr expr;
    ir::Expr extract(ir::Expr e) {
      e.accept(this);
      auto t = expr;
      expr = ir::Expr();
      return t;
    }

    void visit(const ir::Literal* e) {
      expr = e;
    }

    void visit(const ir::Load* e) {
      ir::Expr lastVar = e->loc;
      if (util::contains(currentVars, lastVar)) {
        ir::Expr t = ir::Var::make("t"+to_string(i++), typeOf<double>(), false);
        subExprs->push_back({t,e});
        expr = t;
      }
      else {
        expr = e;
      }
    }

    void visit(const ir::Neg* e) {
      ir::Expr a = extract(e->a);
      if (a == e->a) {
        expr = e;
      }
      else {
        expr = ir::Neg::make(a);
      }
    }

    void visit(const ir::Sqrt* e) {
      ir::Expr a = extract(e->a);
      if (a == e->a) {
        expr = e;
      }
      else {
        expr = ir::Sqrt::make(a);
      }
    }

    void visit(const ir::Add* e) {
      ir::Expr a = extract(e->a);
      ir::Expr b = extract(e->b);
      if (a == e->a && b == e->b) {
        expr = e;
      }
      else {
        expr = ir::Add::make(a, b);
      }
    }

    void visit(const ir::Sub* e) {
      ir::Expr a = extract(e->a);
      ir::Expr b = extract(e->b);
      if (a == e->a && b == e->b) {
        expr = e;
      }
      else {
        expr = ir::Sub::make(a, b);
      }
    }

    void visit(const ir::Mul* e) {
      ir::Expr a = extract(e->a);
      ir::Expr b = extract(e->b);
      if (a == e->a && b == e->b) {
        expr = e;
      }
      else {
        expr = ir::Mul::make(a, b);
      }
    }

    void visit(const ir::Div* e) {
      ir::Expr a = extract(e->a);
      ir::Expr b = extract(e->b);
      if (a == e->a && b == e->b) {
        expr = e;
      }
      else {
        expr = ir::Div::make(a, b);
      }
    }
  };
  return ExtractAvailableExpressions(var, iterators,
                                     schedule, subExprs).extract(expr);
}

}}
