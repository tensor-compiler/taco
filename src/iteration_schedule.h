#ifndef TACO_ITERATION_SCHEDULE_H
#define TACO_ITERATION_SCHEDULE_H

#include <memory>

namespace taco {
struct Expr;

namespace internal {

/// An iteration schedule is a two dimensional ordering of index variables,
/// index traversal paths that describe how to reach non-zero index variable
/// values through tensor indices, and a constraint on each index variable that
/// tells us how to merge tensor index values.
class IterationSchedule {
public:
  IterationSchedule();
  static IterationSchedule make(const taco::Expr& expr);

private:
  struct Content;
  std::shared_ptr<Content> content;
  IterationSchedule(std::shared_ptr<Content> content);
};

}}
#endif
