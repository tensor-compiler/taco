#include "taco/expr.h"

#include "taco/operator.h"
#include "taco/util/name_generator.h"
#include "taco/expr_nodes/expr_nodes.h"

using namespace std;

namespace taco {

// class Var
Var::Var(const std::string& name, Kind kind) : content(new Content) {
  content->name = name;
  content->kind = kind;
}

Var::Var(Kind kind) : Var(util::uniqueName('i'), kind) {
}

std::ostream& operator<<(std::ostream& os, const Var& var) {
  return os << var.getName();
}


// class Expr
Expr::Expr(int val) : Expr(IntImm(val)) {
}

Expr::Expr(float val) : Expr(FloatImm(val)) {
}

Expr::Expr(double val) : Expr(DoubleImm(val)) {
}

Expr Expr::operator-() {
  return Neg(*this);
}

void Expr::accept(expr_nodes::ExprVisitorStrict *v) const {
  ptr->accept(v);
}

std::ostream& operator<<(std::ostream& os, const Expr& expr) {
  if (!expr.defined()) return os << "Expr()";
  expr.ptr->print(os);
  return os;
}

std::vector<TensorBase> getOperands(Expr expr) {
  taco_iassert(expr.defined()) << "Undefined expr";
  struct Visitor : public expr_nodes::ExprVisitor {
    vector<TensorBase> operands;
    virtual void visit(const expr_nodes::ReadNode* op) {
      operands.push_back(op->tensor);
    }
  };
  Visitor visitor;
  expr.accept(&visitor);
  return visitor.operands;
}

}
