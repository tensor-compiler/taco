#include "lower_codegen.h"

#include <set>

#include "internal_tensor.h"
#include "expr.h"
#include "expr_nodes.h"
#include "iterators.h"
#include "iteration_schedule.h"
#include "ir/ir.h"
#include "util/strings.h"
#include "util/collections.h"

using namespace std;
using namespace taco::ir;
using taco::internal::Tensor;

namespace taco {
namespace lower {

ir::Expr lowerToScalarExpression(const taco::Expr& indexExpr,
                                 const Iterators& iterators,
                                 const IterationSchedule& schedule,
                                 const map<Tensor,ir::Expr>& tensorVars,
                                 const map<Tensor,ir::Expr>& temporaries) {

  class ScalarCode : public internal::ExprVisitorStrict {
    using internal::ExprVisitorStrict::visit;

  public:
    const Iterators& iterators;
    const IterationSchedule& schedule;
    const map<Tensor,ir::Expr>& tensorVars;
    const map<Tensor,ir::Expr>& temporaries;
    ScalarCode(const Iterators& iterators,
                 const IterationSchedule& schedule,
                 const map<Tensor,ir::Expr>& tensorVars,
                 const map<Tensor,ir::Expr>& temporaries)
        : iterators(iterators), schedule(schedule), tensorVars(tensorVars),
          temporaries(temporaries) {}

    ir::Expr expr;
    ir::Expr lower(const taco::Expr& indexExpr) {
      indexExpr.accept(this);
      auto e = expr;
      expr = ir::Expr();
      return e;
    }

    void visit(const internal::Read* op) {
      if (util::contains(temporaries, op->tensor)) {
        expr = temporaries.at(op->tensor);
        return;
      }

      storage::Iterator iterator;
      if (op->tensor.getOrder() == 0) {
        iterator = storage::Iterator::makeRoot();
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

  return ScalarCode(iterators,schedule,tensorVars,temporaries).lower(indexExpr);
}

ir::Stmt mergePathIndexVars(ir::Expr var, vector<ir::Expr> pathVars){
  return ir::VarAssign::make(var, ir::Min::make(pathVars));
}

vector<ir::Stmt> printCoordinate(const vector<ir::Expr>& indexVars) {
  vector<string> indexVarNames;
  indexVarNames.reserve((indexVars.size()));
  for (auto& indexVar : indexVars) {
    indexVarNames.push_back(util::toString(indexVar));
  }

  vector<string> fmtstrings(indexVars.size(), "%d");
  string format = util::join(fmtstrings, ",");
  vector<ir::Expr> printvars = indexVars;
  return {ir::Print::make("("+util::join(indexVarNames)+") = "
                          "("+format+")\\n", printvars)};
}

}}
