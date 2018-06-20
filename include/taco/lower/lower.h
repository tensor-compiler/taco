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

/// Lower a concrete index statement to a function in the low-level IR.
/// \arg stmt      A concrete index statement to lower.
/// \arg name      The name of the lowered function.
/// \arg assemble  Whether the lowered function should assemble result indices.
/// \arg compute   Whether the lowered function should compute result values.
ir::Stmt lower(IndexStmt stmt, std::string name, bool assemble, bool compute);

}}
#endif
