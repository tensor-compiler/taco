#include "expr.h"

#include "expr_visitor.h"
#include "operator.h"

namespace taco {

Expr::Expr(int val) : Expr(Imm<int>(val)) {
}

Expr::Expr(double val) : Expr(Imm<double>(val)) {
}

void Expr::accept(internal::ExprVisitor *v) const {
  ptr->accept(v);
}

}
