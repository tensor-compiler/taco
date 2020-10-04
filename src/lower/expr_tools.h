#ifndef TACO_AVAILABLE_EXPRS_H
#define TACO_AVAILABLE_EXPRS_H

#include <vector>

namespace taco {
class IndexVar;
class IndexExpr;

/// Retrieves available sub-expression, which are the maximal sub-expressions
/// whose operands are only indexed by the given index variables.
std::vector<IndexExpr>
getAvailableExpressions(const IndexExpr& expr,
                        const std::vector<IndexVar>& vars);

/// Retrieves the minimal sub-expression that covers all the index variables
IndexExpr getSubExpr(IndexExpr expr, const std::vector<IndexVar>& vars);

/// Retrieves the minimal sub-expression that covers all the index variables
/// DEPRECATED: This is a deprecated function to keep functionality while
///             redesigning the index expression IR with reduction nodes.
IndexExpr getSubExprOld(IndexExpr expr, const std::vector<IndexVar>& vars);

}
#endif
