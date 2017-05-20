#include "taco/expr.h"

#include "taco/util/name_generator.h"
#include "taco/expr_nodes/expr_nodes.h"

using namespace std;

namespace taco {

// class Var
struct Var::Content {
  std::string name;
};

Var::Var() : Var(util::uniqueName('i')) {}

Var::Var(const std::string& name) : content(new Content) {
  content->name = name;
}

std::string Var::getName() const {
  return content->name;
}

bool operator==(const Var& a, const Var& b) {
  return a.content == b.content;
}

bool operator<(const Var& a, const Var& b) {
  return a.content < b.content;
}

std::ostream& operator<<(std::ostream& os, const Var& var) {
  return os << var.getName();
}


// class Expr
Expr::Expr(int val) : Expr(new expr_nodes::IntImmNode(val)) {
}

Expr::Expr(float val) : Expr(new expr_nodes::FloatImmNode(val)) {
}

Expr::Expr(double val) : Expr(new expr_nodes::DoubleImmNode(val)) {
}

Expr Expr::operator-() {
  return new expr_nodes::NegNode(this->ptr);
}

void Expr::accept(expr_nodes::ExprVisitorStrict *v) const {
  ptr->accept(v);
}

std::ostream& operator<<(std::ostream& os, const Expr& expr) {
  if (!expr.defined()) return os << "Expr()";
  expr.ptr->print(os);
  return os;
}


// class Read
Access::Access(const Node* n) : Expr(n) {
}

Access::Access(const TensorBase& tensor, const std::vector<Var>& indices)
    : Access(new Node(tensor, indices)) {
}

const Access::Node* Access::getPtr() const {
  return static_cast<const Node*>(ptr);
}

const TensorBase& Access::getTensor() const {
  return getPtr()->tensor;
}

const std::vector<Var>& Access::getIndexVars() const {
  return getPtr()->indexVars;
}

void Access::operator=(const Expr& expr) {
  auto tensor = getPtr()->tensor;
  taco_uassert(!tensor.getExpr().defined()) << "Cannot reassign " << tensor;
  tensor.setExpr(getIndexVars(), expr);
}


// Operators
Expr operator+(const Expr& lhs, const Expr& rhs) {
  return new expr_nodes::AddNode(lhs, rhs);
}

Expr operator-(const Expr& lhs, const Expr& rhs) {
  return new expr_nodes::SubNode(lhs, rhs);
}

Expr operator*(const Expr& lhs, const Expr& rhs) {
  return new expr_nodes::MulNode(lhs, rhs);
}

Expr operator/(const Expr& lhs, const Expr& rhs) {
  return new expr_nodes::DivNode(lhs, rhs);
}

}
