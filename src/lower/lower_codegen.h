#ifndef TACO_LOWER_UTIL_H
#define TACO_LOWER_UTIL_H

#include <string>
#include <vector>
#include <map>

namespace taco {
class TensorBase;
class IndexExpr;
namespace storage {
class Iterator;
}
namespace ir {
class Stmt;
class Expr;
}

namespace lower {
class IterationSchedule;
class Iterators;

std::tuple<std::vector<ir::Expr>,         // parameters
           std::vector<ir::Expr>,         // results
           std::map<TensorBase,ir::Expr>> // mapping
getTensorVars(const TensorBase&);

/// Lower an index expression to an IR expression that computes the index
/// expression for one point in the iteration space (a scalar computation)
ir::Expr
lowerToScalarExpression(const IndexExpr& indexExpr,
                        const Iterators& iterators,
                        const IterationSchedule& schedule,
                        const std::map<TensorBase,ir::Expr>& temporaries);

/// Emit code to merge several tensor path index variables (using a min)
ir::Stmt mergePathIndexVars(ir::Expr var, std::vector<ir::Expr> pathVars);

ir::Expr min(std::string resultName,
             const std::vector<storage::Iterator>& iterators,
             std::vector<ir::Stmt>* statements);

/// Emit code to print a coordinate
std::vector<ir::Stmt> printCoordinate(const std::vector<ir::Expr>& indexVars);

}}
#endif
