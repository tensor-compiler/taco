#include "ir.h"

namespace taco {
namespace internal {

Expr Literal::make(double val, ComponentType type) {
  Literal *lit = new Literal;
  lit->type = type;
  lit->value = *((int64_t*)(&val));
  return lit;
}

Expr Literal::make(int val) {
  Literal *lit = new Literal;
  lit->type = typeOf<int>();
  lit->value = (int64_t)val;
  return lit;
}

Expr Var::make(std::string name, ComponentType type) {
  Var *var = new Var;
  var->type = type;
  var->name = name;
  return var;
}

// Binary Expressions
// helper
ComponentType max_type(Expr a, Expr b) {
  if (a.type() == b.type()) {
    return a.type();
  } else {
    // if either are double, make it double
    if (a.type() == typeOf<double>() || b.type() == typeOf<double>())
      return typeOf<double>();
    else
      return typeOf<float>();
  }
}

Expr Add::make(Expr a, Expr b) {
  return Add::make(a, b, max_type(a, b));
}

Expr Add::make(Expr a, Expr b, ComponentType type) {
  Add *add = new Add;
  add->type = type;
  add->a = a;
  add->b = b;
  return add;
}

Expr Sub::make(Expr a, Expr b) {
  return Sub::make(a, b, max_type(a, b));
}

Expr Sub::make(Expr a, Expr b, ComponentType type) {
  Sub *sub = new Sub;
  sub->type = type;
  sub->a = a;
  sub->b = b;
  return sub;
}



} // namespace internal
} // namespace tac
