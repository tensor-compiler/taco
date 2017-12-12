#ifndef TACO_SCHEDULE_H
#define TACO_SCHEDULE_H

#include <memory>
#include <vector>

namespace taco {

class IndexVar;
class IndexExpr;


class OperatorSplit {
public:
  OperatorSplit(IndexExpr expr, IndexVar old, IndexVar left, IndexVar right);

  IndexExpr getExpr() const;
  IndexVar getOld() const;
  IndexVar getLeft() const;
  IndexVar getRight() const;

private:
  struct Content;
  std::shared_ptr<Content> content;
};

/// Print an operator split.
std::ostream& operator<<(std::ostream&, const OperatorSplit&);


/// A schedule controls code generation and determines how index expression
/// should be computed.
class Schedule {
public:
  Schedule();

  /// Returns the operator splits in the schedule.
  std::vector<OperatorSplit> getOperatorSplits() const;

  /// Returns the operator splits of `expr`.
  std::vector<OperatorSplit> getOperatorSplits(IndexExpr expr) const;

  /// Add an operator split to the schedule.
  void addOperatorSplit(OperatorSplit split);

  /// Removes operator splits from the schedule.
  void clearOperatorSplits();

private:
  struct Content;
  std::shared_ptr<Content> content;
};

/// Print a schedule.
std::ostream& operator<<(std::ostream&, const Schedule&);

}
#endif
