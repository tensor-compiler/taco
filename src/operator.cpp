#include <vector>

#include "var.h"
#include "operator.h"
#include "expr_nodes.h"
#include "internal_tensor.h"

namespace taco {

// class Read
Read::Read(const Node* n) : Expr(n) {
}

Read::Read(const internal::Tensor& tensor, const std::vector<Var>& indices)
    : Read(new Node(tensor, indices)) {
}

const Read::Node* Read::getPtr() const {
  return static_cast<const Node*>(ptr);
}

const internal::Tensor& Read::getTensor() const {
  return getPtr()->tensor;
}

const std::vector<Var>& Read::getIndexVars() const {
  return getPtr()->indexVars;
}

void Read::assign(Expr expr) {
  auto tensor = getPtr()->tensor;
  uassert(!tensor.getExpr().defined()) << "Cannot reassign " << tensor;

  tensor.setIndexVars(getIndexVars());
  tensor.setExpr(expr);
}

// Oeprators
Add operator+(const Expr& lhs, const Expr& rhs) {
  return Add(lhs, rhs);
}

Sub operator-(const Expr& lhs, const Expr& rhs) {
  return Sub(lhs, rhs);
}

Mul operator*(const Expr& lhs, const Expr& rhs) {
  return Mul(lhs, rhs);
}

Div operator/(const Expr& lhs, const Expr& rhs) {
  return Div(lhs, rhs);
}

}
