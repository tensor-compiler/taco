#include <vector>

#include "operator.h"

namespace taco {

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

}
