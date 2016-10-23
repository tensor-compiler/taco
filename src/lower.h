#ifndef TACO_LOWER_H
#define TACO_LOWER_H

#include "expr.h"
#include "tensor.h"
#include "ir.h"

namespace taco {

namespace is {
class IterationSchedule;
}

namespace internal {

enum class LowerKind {
  Assemble,
  Evaluate,
  AssembleAndEvaluate,
  Print
};

/// Lower the tensor object with a defined expression and an iteration schedule
/// into a statement that evaluates it.
ir::Stmt lower(const internal::Tensor& tensor, LowerKind lowerKind);

}}

#endif
