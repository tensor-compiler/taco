#ifndef TACO_SCHEDULE_H
#define TACO_SCHEDULE_H

#include <memory>

namespace taco {
class IndexExpr;

namespace lower {

/// A schedule is used to control the code generator and the code it emits.
/// Scheduling constructs include:
/// - Operator Split: Split an expression operator at an index variable.
class Schedule {
public:
  Schedule();

  void split(const IndexExpr&);

private:
  struct Content;
  std::shared_ptr<Content> content;
};

}}
#endif
