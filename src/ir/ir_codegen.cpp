#include "ir_codegen.h"

#include "taco/ir/ir.h"
#include "taco/error.h"

namespace taco {
namespace ir {

ir::Stmt compoundStore(ir::Expr arr, ir::Expr loc, ir::Expr val) {
  return Store::make(arr, loc, Add::make(Load::make(arr, loc), val));
}

ir::Stmt compoundAssign(ir::Expr lhs, ir::Expr rhs) {
  return VarAssign::make(lhs, Add::make(lhs, rhs));
}

Expr conjunction(std::vector<Expr> exprs) {
  taco_iassert(exprs.size() > 0) << "No expressions to and";
  Expr conjunction = exprs[0];
  for (size_t i = 1; i < exprs.size(); i++) {
    conjunction = ir::And::make(conjunction, exprs[i]);
  }
  return conjunction;
}

}}
