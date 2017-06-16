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
class IterationSchedule;

enum Property {
  Assemble,
  Compute,
  Print,
  Comment
};

/// Lower the tensor object with a defined expression and an iteration schedule
/// into a statement that evaluates it.
ir::Stmt lower(TensorBase tensor, std::string funcName,
               std::set<Property> properties);

}}
#endif
