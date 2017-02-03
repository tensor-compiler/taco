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
lowerToScalarExpression(const taco::Expr& indexExpr,
                        const Iterators& iterators,
                        const IterationSchedule& schedule,
                        const std::map<internal::Tensor,ir::Expr>& tensorVars);

}}
#endif
