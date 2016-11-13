#ifndef TACO_LOWER_SCALAR_EXPRESSION_H
#define TACO_LOWER_SCALAR_EXPRESSION_H

#include <map>
#include <vector>

#include "ir.h"

namespace taco {
class Var;
class Expr;

namespace internal {
class Tensor;
}

namespace lower {
class IterationSchedule;
class Iterators;
class TensorPathStep;

ir::Expr
lowerScalarExpression(const taco::Expr& indexExpr,
                      const Iterators& iterators,
                      const IterationSchedule& schedule,
                      const std::map<internal::Tensor,ir::Expr>& tensorVars);


/// Extract the sub-expressions that have become available to be computed.
/// These are the sub-expressions where `var` is the variable associated with
/// the last storage dimension of all operands
ir::Expr
extractAvailableExpressions(ir::Expr expr, taco::Var var,
                            const Iterators& iterators,
                            const IterationSchedule& schedule,
                            std::vector<std::pair<ir::Expr,ir::Expr>>*subExprs);


/// Removes the expressions whose ptr variable is not one of the step iterators.
ir::Expr
removeExpressions(ir::Expr expr,
                  const std::vector<TensorPathStep>& steps,
                  const Iterators& iterators);

}}
#endif
