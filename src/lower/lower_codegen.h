#ifndef TACO_LOWER_UTIL_H
#define TACO_LOWER_UTIL_H

#include <string>
#include <vector>
#include <map>

namespace taco {
class TensorVar;
class IndexExpr;
class Iterator;
namespace ir {
class Stmt;
class Expr;
}

namespace lower {
class IterationGraph;
class Iterators;

std::tuple<std::vector<ir::Expr>,        // parameters
           std::vector<ir::Expr>,        // results
           std::map<TensorVar,ir::Expr>> // mapping
getTensorVars(const TensorVar&);

/// Lower an index expression to an IR expression that computes the index
/// expression for one point in the iteration space (a scalar computation)
ir::Expr
lowerToScalarExpression(const IndexExpr& indexExpr,
                        const Iterators& iterators,
                        const IterationGraph& iterationGraph,
                        const std::map<TensorVar,ir::Expr>& temporaries);

/// Emit code to merge several tensor path index variables (using a min)
ir::Stmt mergePathIndexVars(ir::Expr var, std::vector<ir::Expr> pathVars);

ir::Expr min(const std::string resultName,
             const std::vector<Iterator>& iterators,
             std::vector<ir::Stmt>* statements);

std::pair<ir::Expr,ir::Expr>
minWithIndicator(const std::string resultName,
                 const std::vector<Iterator>& iterators,
                 std::vector<ir::Stmt>* statements);

/// Emit code to print a coordinate
std::vector<ir::Stmt> printCoordinate(const std::vector<ir::Expr>& indexVars);

}}
#endif
