#ifndef TACO_LOWER_H
#define TACO_LOWER_H

#include <string>
#include <set>

#include "taco/expr.h"
#include "taco/ir/ir.h"
#include "taco/util/collections.h"

namespace taco {
class TensorBase;

namespace lower {
class Schedule;

enum Property {
  Assemble,
  Compute,
  Print,
  Comment,
  Accumulate  /// Accumulate into the result (+=)
};

/// Lower the tensor object with a defined expression and an iteration schedule
/// into a statement that evaluates it.
ir::Stmt lower(TensorBase tensor, std::string functionName,
               const Schedule& schedule, std::set<Property> properties);

}}
#endif
