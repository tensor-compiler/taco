#ifndef TACO_IR_SIMPLIFY_H
#define TACO_IR_SIMPLIFY_H

#include <map>
#include "taco/ir/ir.h"

namespace taco {
namespace ir {
class Expr;
class Stmt;

/// Simplifies an expression (e.g. by applying algebraic identities).
ir::Expr simplify(const ir::Expr& expr);

/// Simplifies a statement (e.g. by applying constant copy propagation).
ir::Stmt simplify(const ir::Stmt& stmt);

/// Simplifies a statement (e.g. by applying constant copy propagation).
ir::Stmt simplifyEnv(const ir::Stmt& stmt, std::map<Expr, std::string, ExprCompare>& varMap);

}}
#endif
