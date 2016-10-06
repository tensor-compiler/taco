#include "iteration_schedule.h"

using namespace std;

namespace taco {
namespace internal {

struct IterationScheduleInternal {

};

IterationSchedule::IterationSchedule(shared_ptr<IterationScheduleInternal> ptr)
    : ptr(ptr) {
}

IterationSchedule IterationSchedule::make(const taco::Expr& expr) {
  return IterationSchedule(nullptr);
}

}}
