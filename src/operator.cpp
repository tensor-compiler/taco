#include <vector>

#include "operator.h"

namespace taco {

void Read::assign(Expr expr) {
  auto tensor = getPtr()->tensor;
  uassert(!tensor.getExpr().defined()) << "Cannot reassign " << tensor;

  tensor.setIndexVars(getIndexVars());
  tensor.setExpr(expr);
}

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
