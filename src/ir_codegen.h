#ifndef TACO_IR_CODEGEN_H
#define TACO_IR_CODEGEN_H

#include <vector>

namespace taco {

namespace ir {
class Expr;
class Stmt;

/// Add `val` to `arr[loc]`
Stmt compoundStore(Expr arr, Expr loc, Expr val);

/// Add `val` to `var`
Stmt compoundAssign(Expr var, Expr val);

/// Returns a conjunction (and) of `exprs`
Expr conjunction(std::vector<Expr> exprs);

}}
#endif
