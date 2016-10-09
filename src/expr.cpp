#include "expr.h"

namespace taco {

Expr::Expr(int val) : Expr(Imm<int>(val)) {}

Expr::Expr(double val) : Expr(Imm<double>(val)) {}
  
}
