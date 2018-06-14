#ifndef TACO_LOWER_H
#define TACO_LOWER_H

#include <string>
#include <set>

#include "taco/index_notation/index_notation.h"
#include "taco/ir/ir.h"
#include "taco/util/collections.h"

namespace taco {
namespace lower {

enum Property {
  Assemble,
  Compute,
  Print,
  Comment,
  Accumulate  /// Accumulate into the result (+=)
};

/// Lower the tensor object with a defined expression and an iteration schedule
/// into a statement that evaluates it.
ir::Stmt lower(TensorVar tensor, std::string functionName,
               std::set<Property> properties, long long allocSize);

/// Checks whether the an index statement can be lowered to C code.  If the
/// statement cannot be lowered and a `reason` string is provided then it is
/// filled with the a reason.
bool isLowerable(IndexStmt stmt, std::string* reason=nullptr);

}}
#endif
