#ifndef TACO_LOWER_H
#define TACO_LOWER_H

#include <string>
#include <set>
#include <memory>

namespace taco {

class IndexStmt;
class LowererImpl;

namespace ir {
class Stmt;
}


/// A lowerer lowers concrete index notation statements according to the given
/// lowerer implementation. See lowerer_impl.h for information about how to
/// create a custom lowerer.
class Lowerer {
public:

  /// Construct a default lowerer that lowers to imperative multi-threaded code.
  Lowerer();

  /// Construct a lowerer that lowers as specified by the lowerer impl.  The
  /// lowerer will delete the impl object.
  Lowerer(LowererImpl* impl);

  /// Retrieve the lowerer implementation.
  std::shared_ptr<LowererImpl> getLowererImpl();

private:
  std::shared_ptr<LowererImpl> impl;
};


/// Lower a concrete index statement to a function in the low-level IR.  You may
/// specify whether the lowered function should assemble, compute, or both
/// (by default it both assembles and computes) and you may provide a lowerer
/// that specifies how to lower different parts of a concrete index notation
/// statement.
ir::Stmt lower(IndexStmt stmt, std::string name,
               bool assemble=true, bool compute=true,
               Lowerer lowerer=Lowerer());

/// Checks whether the an index statement can be lowered to C code.  If the
/// statement cannot be lowered and a `reason` string is provided then it is
/// filled with the a reason.
bool isLowerable(IndexStmt stmt, std::string* reason=nullptr);


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
