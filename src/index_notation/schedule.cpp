#include "taco/index_notation/schedule.h"

#include <map>

#include "taco/index_notation/index_notation.h"
#include "taco/util/collections.h"
#include "taco/util/strings.h"

using namespace std;

namespace taco {

// class Workspace
struct Workspace::Content {
  IndexExpr expr;
  IndexVar i;
  IndexVar iw;
  TensorVar workspace;
};

Workspace::Workspace() : content(nullptr) {
}

Workspace::Workspace(IndexExpr expr, IndexVar i, IndexVar iw,
                     TensorVar workspace) : content(new Content) {
  content->expr = expr;
  content->i = i;
  content->iw = iw;
  content->workspace = workspace;
}

IndexExpr Workspace::getExpr() const {
  return content->expr;
}

IndexVar Workspace::geti() const {
  return content->i;
}

IndexVar Workspace::getiw() const {
  return content->iw;
}

TensorVar Workspace::getWorkspace() const {
  return content->workspace;
}

bool Workspace::defined() const {
  return content != nullptr;
}

std::ostream& operator<<(std::ostream& os, const Workspace& workspace) {
  return os << workspace.getExpr() << ": " << workspace.geti() << ", "
            << workspace.getiw() << ", " << workspace.getWorkspace();
}


// class Schedule
struct Schedule::Content {
  map<IndexExpr, Workspace> workspaces;
};

Schedule::Schedule() : content(new Content) {
}

std::vector<Workspace> Schedule::getWorkspaces() const {
  vector<Workspace> workspaces;
  for (auto& workspace : content->workspaces) {
    workspaces.push_back(workspace.second);
  }
  return workspaces;
}

Workspace Schedule::getWorkspace(IndexExpr expr) const {
  if (!util::contains(content->workspaces, expr)) {
    return Workspace();
  }
  return content->workspaces.at(expr);
}

void Schedule::addWorkspace(Workspace workspace) {
  if (!util::contains(content->workspaces, workspace.getExpr())) {
    content->workspaces.insert({workspace.getExpr(), workspace});
  }
  else {
    content->workspaces.at(workspace.getExpr()) = workspace;
  }
}

void Schedule::clearWorkspaces() {
  content->workspaces.clear();
}

std::ostream& operator<<(std::ostream& os, const Schedule& schedule) {
  auto workspaces = schedule.getWorkspaces();
  if (workspaces.size() > 0) {
    os << "Workspace Commands:" << endl << util::join(workspaces, "\n");
  }
  return os;
}

}
