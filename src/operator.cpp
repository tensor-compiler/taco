#include <vector>

#include "taco/tensor_base.h"
#include "taco/operator.h"
#include "taco/expr_nodes/expr_nodes.h"

namespace taco {

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
