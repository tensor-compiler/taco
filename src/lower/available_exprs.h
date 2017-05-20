#ifndef TACO_AVAILABLE_EXPRS_H
#define TACO_AVAILABLE_EXPRS_H

#include <vector>

namespace taco {
class IndexVar;
class IndexExpr;

namespace lower {

/// Retrieves available sub-expression, which are the maximal sub-expressions
/// whose operands are only indexed by the given index variables.
std::vector<IndexExpr>
getAvailableExpressions(const IndexExpr& expr,
                        const std::vector<IndexVar>& vars);

}}
#endif
