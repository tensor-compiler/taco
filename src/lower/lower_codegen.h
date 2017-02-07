#ifndef TACO_LOWER_UTIL_H
#define TACO_LOWER_UTIL_H

#include <vector>
#include <map>

namespace taco {
class Expr;

namespace internal {
class Tensor;
}

namespace ir {
class Stmt;
class Expr;
}

namespace lower {
class IterationSchedule;
class Iterators;

/// Lower an index expression to an IR expression that computes the index
/// expression for one point in the iteration space (a scalar computation)
ir::Expr
lowerToScalarExpression(const taco::Expr& indexExpr,
                        const Iterators& iterators,
                        const IterationSchedule& schedule,
                        const std::map<internal::Tensor,ir::Expr>& tensorVars);

/// Emit code to merge several tensor path index variables (using a min)
ir::Stmt mergePathIndexVars(ir::Expr var, std::vector<ir::Expr> pathVars);

/// Emit code to print a coordinate
std::vector<ir::Stmt> printCoordinate(const std::vector<ir::Expr>& indexVars);

}}
#endif
