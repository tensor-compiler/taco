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


/// A `Lowerer` lowers concrete index notation statements as specified by a
/// `LowererImpl`.  The default `Lowerer`/`LowererImpl` lowers to sequential,
/// multithreaded, GPU, and vectorized code as specified by the concrete index
/// notation.  `LowererImpl`, however, can be extended and it's methods
/// overridden to insert custom lowering code to e.g. target specialized
/// hardware.  See `lowerer_impl.h` for information about how to create custom
/// lowerers.
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


/// Lower a concrete index notation statement to a function in the low-level
/// imperative IR.  You may specify whether the lowered function should
/// assemble result indices, compute result values, or both. You may optionally
/// also provide a custom `Lowerer` to specify custom ways to lower some or all
/// parts of a concrete index notation statement.
ir::Stmt lower(IndexStmt stmt, std::string functionName,
               bool assemble=true, bool compute=true,
               Lowerer lowerer=Lowerer());

/// Check whether the an index statement can be lowered to C code.  If the
/// statement cannot be lowered and a `reason` string is provided then it is
/// filled with the a reason.
bool isLowerable(IndexStmt stmt, std::string* reason=nullptr);

}
#endif
