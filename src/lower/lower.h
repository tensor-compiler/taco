#ifndef TACO_LOWER_H
#define TACO_LOWER_H

#include <string>
#include <set>

#include "expr.h"
#include "tensor.h"
#include "ir.h"
#include "util/collections.h"

namespace taco {
namespace lower {
class IterationSchedule;

enum Property {
  Assemble,
  Compute,
  Print
};

/// Lower the tensor object with a defined expression and an iteration schedule
/// into a statement that evaluates it.
ir::Stmt lower(const internal::Tensor& tensor,
               std::string funcName, const std::set<Property>& properties);

}}
#endif
