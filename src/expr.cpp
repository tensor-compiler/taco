#include "expr.h"

namespace taco {

Var::Kind Var::Free = Var::Kind::Free;
Var::Kind Var::Reduction = Var::Kind::Reduction;

Expr::Expr(int val) : Expr(Imm<int>(val)) {}

Expr::Expr(double val) : Expr(Imm<double>(val)) {}
  
}
