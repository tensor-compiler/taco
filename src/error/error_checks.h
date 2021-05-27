#ifndef TACO_ERROR_CHECKS_H
#define TACO_ERROR_CHECKS_H

#include <vector>
#include <string>
#include <tuple>

namespace taco {
class IndexVar;
class IndexExpr;
class Format;
class Shape;

namespace error {

/// Check whether all dimensions indexed by the same variable are the same.
/// If they are not, then the first element of the returned tuple will be false,
/// and a human readable error will be returned in the second component.
std::pair<bool, std::string> dimensionsTypecheck(const std::vector<IndexVar>& resultVars,
                                            const IndexExpr& expr,
                                            const Shape& shape);

/// Returns true iff the index expression contains a transposition.
bool containsTranspose(const Format& resultFormat,
                       const std::vector<IndexVar>& resultVars,
                       const IndexExpr& expr);

}}
#endif
