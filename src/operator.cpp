#include <vector>

#include "operator.h"

namespace taco {

void Read::assign(Expr expr) {
  auto tensor = getPtr()->tensor;
  uassert(!tensor.getExpr().defined()) << "Cannot reassign " << tensor;

  tensor.setIndexVars(getIndexVars());
  tensor.setExpr(expr);
}

Add operator+(const Expr& lhs, const Expr& rhs) {
  std::vector<Expr> operands;

  if (isa<Add>(lhs)) {
    Add addLhs = to<Add>(lhs);
    operands.insert(operands.end(), addLhs.getPtr()->operands.begin(),
                    addLhs.getPtr()->operands.end());
  } else {
    operands.push_back(lhs);
  }

  if (isa<Add>(rhs)) {
    Add addRhs = to<Add>(rhs);
    operands.insert(operands.end(), addRhs.getPtr()->operands.begin(),
                    addRhs.getPtr()->operands.end());
  } else {
    operands.push_back(rhs);
  }

  return Add(operands);
}

Sub operator-(const Expr& lhs, const Expr& rhs) {
  return Sub(lhs, rhs);
}

Mul operator*(const Expr& lhs, const Expr& rhs) {
  std::vector<Expr> operands;
  
  if (isa<Mul>(lhs)) {
    Mul mulLhs = to<Mul>(lhs);
    operands.insert(operands.end(), mulLhs.getPtr()->operands.begin(),
                    mulLhs.getPtr()->operands.end());
  } else {
    operands.push_back(lhs);
  }

  if (isa<Mul>(rhs)) {
    Mul mulRhs = to<Mul>(rhs);
    operands.insert(operands.end(), mulRhs.getPtr()->operands.begin(),
                    mulRhs.getPtr()->operands.end());
  } else {
    operands.push_back(rhs);
  }

  return Mul(operands);
}

Div operator/(const Expr& lhs, const Expr& rhs) {
  return Div(lhs, rhs);
}

}
