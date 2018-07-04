#ifndef TACO_LOWER_H
#define TACO_LOWER_H

#include <string>


namespace taco {

class IndexStmt;

namespace ir {
class Stmt;
}

/// Lower a concrete index statement to a function in the low-level IR.
/// \arg stmt      A concrete index statement to lower.
/// \arg name      The name of the lowered function.
/// \arg assemble  Whether the lowered function should assemble result indices.
/// \arg compute   Whether the lowered function should compute result values.
ir::Stmt lower(IndexStmt stmt, std::string name, bool assemble, bool compute);

/// Checks whether the an index statement can be lowered to C code.  If the
/// statement cannot be lowered and a `reason` string is provided then it is
/// filled with the a reason.
bool isLowerable(IndexStmt stmt, std::string* reason=nullptr);

/// Prints the hierarchy of merge cases that result from lowering `stmt`.
void printMergeCaseHierarchy(IndexStmt stmt, std::ostream& os);

}


#include <set>
namespace taco {

class Assignment;
namespace old {
enum Property {
  Assemble,
  Compute,
  Print,
  Comment,
  Accumulate  /// Accumulate into the result (+=)
};

/// Lower the tensor object with a defined expression and an iteration schedule
/// into a statement that evaluates it.
ir::Stmt lower(Assignment assignment, std::string functionName,
               std::set<Property> properties, long long allocSize);
}}
#endif
