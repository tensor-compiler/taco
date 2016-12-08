#ifndef TACO_LOWER_UTIL_H
#define TACO_LOWER_UTIL_H

#include <vector>

namespace taco {

namespace ir {
class Stmt;
class Expr;
}

namespace lower {

/// Emit code to merge several tensor path index variables (using a min)
ir::Stmt mergePathIndexVars(ir::Expr var, std::vector<ir::Expr> pathVars);

/// Emit code to print a coordinate
std::vector<ir::Stmt> printCoordinate(const std::vector<ir::Expr>& indexVars);

}}
#endif
