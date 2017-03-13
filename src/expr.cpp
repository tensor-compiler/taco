#include "taco/expr.h"

#include "taco/operator.h"

namespace taco {

Expr::Expr(int val) : Expr(IntImm(val)) {
}

Expr::Expr(float val) : Expr(FloatImm(val)) {
}

Expr::Expr(double val) : Expr(DoubleImm(val)) {
}

Expr Expr::operator-() {
  return Neg(*this);
}

void Expr::accept(internal::ExprVisitorStrict *v) const {
  ptr->accept(v);
}

std::ostream& operator<<(std::ostream& os, const Expr& expr) {
  if (!expr.defined()) return os << "Expr()";
  expr.ptr->print(os);
  return os;
}

}
