#ifndef TACO_ITERATION_SCHEDULE_H
#define TACO_ITERATION_SCHEDULE_H

#include <memory>

namespace taco {
struct Expr;

namespace internal {
struct IterationScheduleInternal;

class IterationSchedule {
public:
  static IterationSchedule make(const taco::Expr& expr);

private:
  std::shared_ptr<IterationScheduleInternal> ptr;
  IterationSchedule(std::shared_ptr<IterationScheduleInternal> ptr);
};

}}
#endif
