#ifndef TACO_ERROR_CHECKS_H
#define TACO_ERROR_CHECKS_H

#include <vector>

namespace taco {
class IndexVar;
class IndexExpr;
class Format;

namespace error {

/// Returns true iff the index expression contains a transposition.
bool containsTranspose(const Format& resultFormat,
                       const std::vector<IndexVar>& resultVars,
                       const IndexExpr& expr);

/// Returns true iff the index expression distributes values over result
/// dimensions.
bool containsDistribution(const std::vector<IndexVar>& resultVars,
                          const IndexExpr& expr);

}}
#endif
