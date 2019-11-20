#ifndef TACO_IR_CODEGEN_H
#define TACO_IR_CODEGEN_H

#include <vector>
#include "taco/ir_tags.h"

namespace taco {

namespace ir {
class Expr;
class Stmt;

/// Generate `a[i] += val;`
Stmt compoundStore(Expr a, Expr i, Expr val, bool use_atomics=false, PARALLEL_UNIT atomic_parallel_unit=PARALLEL_UNIT::NOT_PARALLEL);

/// Generate `a += val;`
Stmt compoundAssign(Expr a, Expr val, bool use_atomics=false, PARALLEL_UNIT atomic_parallel_unit=PARALLEL_UNIT::NOT_PARALLEL);

/// Generate `exprs_0 && ... && exprs_n`
Expr conjunction(std::vector<Expr> exprs);

/// Generate a statement that doubles the size of `a` if it is full (loc cannot 
/// be written to).
Stmt doubleSizeIfFull(Expr a, Expr size, Expr loc);

/// Generate a statement that resizes `a` to be double its original size or at 
/// least equal to `loc` if it is full (loc cannot be written to).
Stmt atLeastDoubleSizeIfFull(Expr a, Expr size, Expr loc);

}}
#endif
