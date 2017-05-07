#ifndef TACO_IR_SIMPLIFY_H
#define TACO_IR_SIMPLIFY_H

namespace taco {
namespace ir {
class Expr;
class Stmt;

/// Simplifies an expression (e.g. by applying algebraic identities).
ir::Expr simplify(const ir::Expr& expr);

/// Simplifies a statement (e.g. by applying constant copy propagation).
ir::Stmt simplify(const ir::Stmt& stmt);

}}
#endif
