#ifndef TACO_SCHEDULE_H
#define TACO_SCHEDULE_H

#include <memory>
#include <vector>
#include <string>
#include <ostream>

namespace taco {

class IndexExpr;

class Reorder;
class Precompute;

/// A schedule controls code generation and determines how index expression
/// should be computed.
class Schedule {
public:
  Schedule();

  /// Returns the workspace commands in the schedule.
  std::vector<Precompute> getPrecomputes() const;

  /// Returns the workspace of `expr`.  The result is undefined if `expr` is not
  /// stored to a workspace.
  Precompute getPrecompute(IndexExpr expr) const;

  /// Add a workspace command to the schedule.
  void addPrecompute(Precompute precompute);

  /// Removes workspace commands from the schedule.
  void clearPrecomputes();

private:
  struct Content;
  std::shared_ptr<Content> content;
};

/// Print a schedule.
std::ostream& operator<<(std::ostream&, const Schedule&);

}
#endif
