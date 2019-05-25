#include "ir_generators.h"

#include "taco/ir/ir.h"
#include "taco/error.h"
#include "taco/util/strings.h"

namespace taco {
namespace ir {

Stmt compoundStore(Expr a, Expr i, Expr val) {
  Expr add = (val.type().getKind() == Datatype::Bool) 
             ? Or::make(Load::make(a, i), val)
             : Add::make(Load::make(a, i), val);
  return Store::make(a, i, add);
}

Stmt compoundAssign(Expr a, Expr val) {
  Expr add = (val.type().getKind() == Datatype::Bool) 
             ? Or::make(a, val) : Add::make(a, val);
  return Assign::make(a, add);
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
  Stmt realloc = Allocate::make(a, Mul::make(size, 2), true, size);
  Stmt resize = Assign::make(size, Mul::make(size, 2));
  Stmt ifBody = Block::make({realloc, resize});
  return IfThenElse::make(Lte::make(size, needed), ifBody);
}

Stmt atLeastDoubleSizeIfFull(Expr a, Expr size, Expr needed) {
  Expr newSizeVar = Var::make(util::toString(a) + "_new_size", Int());
  Expr newSize = Max::make(Mul::make(size, 2), Add::make(needed, 1));
  Stmt computeNewSize = VarDecl::make(newSizeVar, newSize);
  Stmt realloc = Allocate::make(a, newSizeVar, true, size);
  Stmt updateSize = Assign::make(size, newSizeVar);
  Stmt ifBody = Block::make({computeNewSize, realloc, updateSize});
  return IfThenElse::make(Lte::make(size, needed), ifBody);
}

}}
