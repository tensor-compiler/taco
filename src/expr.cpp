#include "expr.h"

#include "expr_visitor.h"
#include "operator.h"

namespace taco {

Expr::Expr(int val) : Expr(IntImm(val)) {
}

Expr::Expr(float val) : Expr(FloatImm(val)) {
}

Expr::Expr(double val) : Expr(DoubleImm(val)) {
}

void Expr::accept(internal::ExprVisitor *v) const {
  ptr->accept(v);
}

std::ostream& operator<<(std::ostream& os, const Expr& expr) {
  expr.ptr->print(os);
  return os;
}

}
