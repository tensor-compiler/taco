#include "ir.h"

namespace tac {
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

} // namespace internal
} // namespace tac
