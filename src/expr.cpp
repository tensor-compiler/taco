#include "expr.h"

namespace taco {

VarNode::VarNode(Kind kind, const std::string& name) : kind(kind), name(name) {}

Var::Var(Kind kind, const std::string& name) : Var(new Node(kind, name)) {}

Expr::Expr(int val) : Expr(Imm<int>(val)) {}

Expr::Expr(double val) : Expr(Imm<double>(val)) {}
  
}
