#ifndef TACO_MODE_ACCESS_H
#define TACO_MODE_ACCESS_H

#include "taco/util/comparable.h"
#include "taco/index_notation/index_notation.h"

namespace taco {

/// The access of a single mode in an `Access` expression.  For example, the
/// expression `A(i,j)` consist of two mode accesses, A(1) and A(2).
class ModeAccess : public util::Comparable<ModeAccess> {
public:

  /// Create a mode access.
  ModeAccess(Access access, int mode);

  /// Get the `Access` expression of the mode access.
  Access getAccess() const;

  /// Get the position of the accessed mode.
  size_t getModePos() const;

private:
  Access access;
  int mode;
};

bool operator==(const ModeAccess&, const ModeAccess&);
bool operator<(const ModeAccess&, const ModeAccess&);
std::ostream &operator<<(std::ostream &os, const ModeAccess &);

}
#endif
