#include "taco/index_notation/schedule.h"

#include <map>

#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/transformations.h"
#include "taco/util/collections.h"
#include "taco/util/strings.h"
#include "taco/error/error_messages.h"

using namespace std;

namespace taco {

// class Schedule
struct Schedule::Content {
  map<IndexExpr, Precompute> precomputes;
};

Schedule::Schedule() : content(new Content) {
}

std::vector<Precompute> Schedule::getPrecomputes() const {
  vector<Precompute> workspaces;
  for (auto& workspace : content->precomputes) {
    workspaces.push_back(workspace.second);
  }
  return workspaces;
}

Precompute Schedule::getPrecompute(IndexExpr expr) const {
  if (!util::contains(content->precomputes, expr)) {
    return Precompute();
  }
  return content->precomputes.at(expr);
}

void Schedule::addPrecompute(Precompute workspace) {
  if (!util::contains(content->precomputes, workspace.getExpr())) {
    content->precomputes.insert({workspace.getExpr(), workspace});
  }
  else {
    content->precomputes.at(workspace.getExpr()) = workspace;
  }
}

void Schedule::clearPrecomputes() {
  content->precomputes.clear();
}

std::ostream& operator<<(std::ostream& os, const Schedule& schedule) {
  auto workspaces = schedule.getPrecomputes();
  if (workspaces.size() > 0) {
    os << "Workspace Commands:" << endl << util::join(workspaces, "\n");
  }
  return os;
}

}
