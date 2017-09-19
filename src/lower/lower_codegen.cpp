#include "lower_codegen.h"

#include <set>

#include "taco/tensor.h"
#include "taco/expr.h"
#include "taco/expr_nodes/expr_nodes.h"
#include "iterators.h"
#include "iteration_schedule.h"
#include "taco/ir/ir.h"
#include "taco/util/strings.h"
#include "taco/util/collections.h"

using namespace std;
using namespace taco::ir;
using namespace taco::expr_nodes;

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
  ir::Expr tensorVar = ir::Var::make(tensor.getName(), Type(Type::Float,64),
                                     tensor.getFormat());
  mapping.insert({tensor, tensorVar});
  results.push_back(tensorVar);

  // Pack operand tensors into input parameter list
  vector<TensorBase> operands = expr_nodes::getOperands(tensor.getExpr());
  for (TensorBase& operand : operands) {
    taco_iassert(!util::contains(mapping, operand));
    ir::Expr operandVar = ir::Var::make(operand.getName(), Type(Type::Float,64),
                                        operand.getFormat());
    mapping.insert({operand, operandVar});
    parameters.push_back(operandVar);
  }

  return std::tuple<std::vector<ir::Expr>, std::vector<ir::Expr>,
      std::map<TensorBase,ir::Expr>> {parameters, results, mapping};
}

ir::Expr lowerToScalarExpression(const IndexExpr& indexExpr,
                                 const Iterators& iterators,
                                 const IterationSchedule& schedule,
                                 const map<TensorBase,ir::Expr>& temporaries) {

  class ScalarCode : public expr_nodes::ExprVisitorStrict {
    using expr_nodes::ExprVisitorStrict::visit;

  public:
    const Iterators& iterators;
    const IterationSchedule& schedule;
    const map<TensorBase,ir::Expr>& temporaries;
    ScalarCode(const Iterators& iterators,
                 const IterationSchedule& schedule,
                 const map<TensorBase,ir::Expr>& temporaries)
        : iterators(iterators), schedule(schedule), temporaries(temporaries) {}

    ir::Expr expr;
    ir::Expr lower(const IndexExpr& indexExpr) {
      indexExpr.accept(this);
      auto e = expr;
      expr = ir::Expr();
      return e;
    }

    void visit(const AccessNode* op) {
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

    void visit(const NegNode* op) {
      expr = ir::Neg::make(lower(op->a));
    }

    void visit(const SqrtNode* op) {
      expr = ir::Sqrt::make(lower(op->a));
    }

    void visit(const AddNode* op) {
      expr = ir::Add::make(lower(op->a), lower(op->b));
    }

    void visit(const SubNode* op) {
      expr = ir::Sub::make(lower(op->a), lower(op->b));
    }

    void visit(const MulNode* op) {
      expr = ir::Mul::make(lower(op->a), lower(op->b));
    }

    void visit(const DivNode* op) {
      expr = ir::Div::make(lower(op->a), lower(op->b));
    }

    void visit(const IntImmNode* op) {
      expr = ir::Expr(op->val);
    }

    void visit(const FloatImmNode* op) {
      expr = ir::Expr(op->val);
    }

    void visit(const DoubleImmNode* op) {
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
  taco_iassert(iterators.size() > 0);
  taco_iassert(statements != nullptr);
  ir::Expr minVar;
  if (iterators.size() > 1) {
    minVar = ir::Var::make(resultName, Type(Type::Int));
    ir::Expr minExpr = ir::Min::make(getIdxVars(iterators));
    ir::Stmt initIdxStmt = ir::VarAssign::make(minVar, minExpr, true);
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
