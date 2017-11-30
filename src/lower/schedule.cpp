#include "taco/lower/schedule.h"

#include "taco/expr.h"

namespace taco {
namespace lower {

// class Schedule
struct Schedule::Content {
  std::vector<IndexExpr> splits;
};

Schedule::Schedule() : content(new Content) {
}

void Schedule::split(const IndexExpr& indexExpr) {
  content->splits.push_back(indexExpr);
}

}}
