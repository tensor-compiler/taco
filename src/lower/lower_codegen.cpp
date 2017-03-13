#include "lower_codegen.h"

#include <set>

#include "taco/tensor_base.h"
#include "taco/expr.h"
#include "taco/expr_nodes/expr_nodes.h"
#include "iterators.h"
#include "iteration_schedule.h"
#include "ir/ir.h"
#include "taco/util/strings.h"
#include "taco/util/collections.h"

using namespace std;
using namespace taco::ir;

namespace taco {
namespace lower {

std::tuple<std::vector<ir::Expr>,         // parameters
           std::vector<ir::Expr>,         // results
           std::map<TensorBase,ir::Expr>> // mapping
getTensorVars(const TensorBase& tensor) {
  vector<ir::Expr> parameters;
  vector<ir::Expr> results;
  map<TensorBase, ir::Expr> mapping;

  // Pack result tensor into output parameter list
  ir::Expr tensorVar = ir::Var::make(tensor.getName(), typeOf<double>(),
                                     tensor.getFormat());
  mapping.insert({tensor, tensorVar});
  results.push_back(tensorVar);

  // Pack operand tensors into input parameter list
  vector<TensorBase> operands = internal::getOperands(tensor.getExpr());
  for (TensorBase& operand : operands) {
    iassert(!util::contains(mapping, operand));
    ir::Expr operandVar = ir::Var::make(operand.getName(), typeOf<double>(),
                                        operand.getFormat());
    mapping.insert({operand, operandVar});
    parameters.push_back(operandVar);
  }

  return std::tuple<std::vector<ir::Expr>, std::vector<ir::Expr>,
      std::map<TensorBase,ir::Expr>> {parameters, results, mapping};
}

ir::Expr lowerToScalarExpression(const taco::Expr& indexExpr,
                                 const Iterators& iterators,
                                 const IterationSchedule& schedule,
                                 const map<TensorBase,ir::Expr>& temporaries) {

  class ScalarCode : public internal::ExprVisitorStrict {
    using internal::ExprVisitorStrict::visit;

  public:
    const Iterators& iterators;
    const IterationSchedule& schedule;
    const map<TensorBase,ir::Expr>& temporaries;
    ScalarCode(const Iterators& iterators,
                 const IterationSchedule& schedule,
                 const map<TensorBase,ir::Expr>& temporaries)
        : iterators(iterators), schedule(schedule), temporaries(temporaries) {}

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
      TensorPath path = schedule.getTensorPath(op);
      storage::Iterator iterator = (op->tensor.getOrder() == 0)
          ? iterators.getRoot(path)
          : iterators[path.getLastStep()];
      ir::Expr ptr = iterator.getPtrVar();
      ir::Expr values = GetProperty::make(iterator.getTensor(),
                                          TensorProperty::Values);
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
  return ScalarCode(iterators,schedule,temporaries).lower(indexExpr);
}

ir::Stmt mergePathIndexVars(ir::Expr var, vector<ir::Expr> pathVars){
  return ir::VarAssign::make(var, ir::Min::make(pathVars));
}

ir::Expr min(std::string resultName,
             const std::vector<storage::Iterator>& iterators,
             std::vector<Stmt>* statements) {
  iassert(iterators.size() > 0);
  iassert(statements != nullptr);
  ir::Expr minVar;
  if (iterators.size() > 1) {
    minVar = ir::Var::make(resultName, typeOf<int>(), false);
    ir::Expr minExpr = ir::Min::make(getIdxVars(iterators));
    ir::Stmt initIdxStmt = ir::VarAssign::make(minVar, minExpr);
    statements->push_back(initIdxStmt);
  }
  else {
    minVar = iterators[0].getIdxVar();
  }
  return minVar;
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
