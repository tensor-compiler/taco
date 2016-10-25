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
    : Read(new internal::ReadNode(tensor, indices)) {
}

const Read::Node* Read::getPtr() const {
  return static_cast<const Read::Node*>(Read::ptr);
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


// Merge
template <typename T>
std::vector<Expr> mergeOperands(const Expr& lhs, const Expr& rhs) {
  std::vector<Expr> operands;

  if (isa<T>(lhs)) {
    T tLhs = to<T>(lhs);
    operands.insert(operands.end(), tLhs.getPtr()->operands.begin(),
                    tLhs.getPtr()->operands.end());
  } else {
    operands.push_back(lhs);
  }

  if (isa<T>(rhs)) {
    T tRhs = to<T>(rhs);
    operands.insert(operands.end(), tRhs.getPtr()->operands.begin(),
                    tRhs.getPtr()->operands.end());
  } else {
    operands.push_back(rhs);
  }

  return operands;
}

// Operators
Add operator+(const Expr& lhs, const Expr& rhs) {
  return Add(mergeOperands<Add>(lhs, rhs));
}

Sub operator-(const Expr& lhs, const Expr& rhs) {
  return Sub(lhs, rhs);
}

Mul operator*(const Expr& lhs, const Expr& rhs) {
  return Mul(mergeOperands<Mul>(lhs, rhs));
}

Div operator/(const Expr& lhs, const Expr& rhs) {
  return Div(lhs, rhs);
}

}
