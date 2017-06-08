#ifndef TACO_ERROR_CHECKS_H
#define TACO_ERROR_CHECKS_H

#include <vector>
#include <string>

namespace taco {
class IndexVar;
class IndexExpr;
class Format;

namespace error {

/// Check that the dimensions indexed by the same variable are the same
bool dimensionsTypecheck(const std::vector<IndexVar>& resultVars,
                         const IndexExpr& expr,
                         const std::vector<int>& dimensions);

/// Returns error strings for index variables that don't typecheck
std::string dimensionTypecheckErrors(const std::vector<IndexVar>& resultVars,
                                     const IndexExpr& expr,
                                     const std::vector<int>& dimensions);

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
