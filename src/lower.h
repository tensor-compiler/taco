#ifndef TACO_LOWER_H
#define TACO_LOWER_H

#include <string>
#include <vector>
#include <set>

#include "expr.h"
#include "tensor.h"
#include "ir.h"
#include "util/collections.h"

namespace taco {

namespace is {
class IterationSchedule;
}

namespace internal {

enum Property {
  Assemble,
  Evaluate,
  Print
};

/// Lower the tensor object with a defined expression and an iteration schedule
/// into a statement that evaluates it.
ir::Stmt lower(const internal::Tensor& tensor,
               const std::vector<Property>& properties,
               std::string funcName);

}}

#endif
