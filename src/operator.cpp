#include <vector>

#include "taco/tensor_base.h"
#include "taco/operator.h"
#include "taco/expr_nodes.h"

namespace taco {

// class Read
Read::Read(const Node* n) : Expr(n) {
}

Read::Read(const TensorBase& tensor) : Read(tensor, {}) {
}

Read::Read(const TensorBase& tensor, const std::vector<Var>& indices)
    : Read(new Node(tensor, indices)) {
}

const Read::Node* Read::getPtr() const {
  return static_cast<const Node*>(ptr);
}

const TensorBase& Read::getTensor() const {
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

// Operators
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
