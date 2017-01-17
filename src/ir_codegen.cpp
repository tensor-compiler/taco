#include "ir_codegen.h"

#include "ir.h"

namespace taco {
namespace ir {

ir::Stmt compoundStore(ir::Expr arr, ir::Expr loc, ir::Expr val) {
  return Store::make(arr, loc, Add::make(Load::make(arr, loc), val));
}

ir::Stmt compoundAssign(ir::Expr lhs, ir::Expr rhs) {
  return VarAssign::make(lhs, Add::make(lhs, rhs));
}

}}
