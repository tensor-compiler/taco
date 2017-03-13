#include "taco/expr.h"

#include "taco/operator.h"
#include "taco/util/name_generator.h"

namespace taco {

// class Var
Var::Var(const std::string& name, Kind kind) : content(new Content) {
  content->name = name;
  content->kind = kind;
}

Var::Var(Kind kind) : Var(util::uniqueName('i'), kind) {
}

std::ostream& operator<<(std::ostream& os, const Var& var) {
  return os << (var.getKind() == Var::Sum ? "+" : "") << var.getName();
}


// class Expr
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
