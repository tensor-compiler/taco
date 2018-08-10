#include "ir_generators.h"

#include "taco/ir/ir.h"
#include "taco/error.h"

namespace taco {
namespace ir {

Stmt compoundStore(Expr a, Expr i, Expr val) {
  return Store::make(a, i, Add::make(Load::make(a, i), val));
}

Stmt compoundAssign(Expr a, Expr val) {
  return Assign::make(a, Add::make(a, val));
}

Expr conjunction(std::vector<Expr> exprs) {
  taco_iassert(exprs.size() > 0) << "No expressions to and";
  Expr conjunction = exprs[0];
  for (size_t i = 1; i < exprs.size(); i++) {
    conjunction = And::make(conjunction, exprs[i]);
  }
  return conjunction;
}

Stmt doubleSizeIfFull(Expr a, Expr size, Expr needed) {
  Stmt resize = Assign::make(size, Mul::make(size, 2));
  Stmt realloc = Allocate::make(a, size, true);
  Stmt ifBody = Block::make({resize, realloc});
  return IfThenElse::make(Lte::make(size,needed), ifBody);
}

}}
