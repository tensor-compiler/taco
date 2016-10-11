#include "iteration_schedule.h"

#include <set>
#include <vector>
#include <queue>

#include "var.h"
#include "expr.h"
#include "operator.h"
#include "expr_visitor.h"

#include "util/strings.h"
#include "util/collections.h"

using namespace std;

namespace taco {
namespace internal {

// class TensorPath
struct TensorPath::Content {
  Content(Tensor tensor, vector<Var> path) : tensor(tensor), path(path) {
  }
  Tensor tensor;
  vector<Var> path;
};

TensorPath::TensorPath(Tensor tensor, vector<Var> path)
    : content(new TensorPath::Content(tensor, path)) {
}

const Tensor& TensorPath::getTensor() const {
  return content->tensor;
}

const std::vector<Var>& TensorPath::getPath() const {
  return content->path;
}

std::ostream& operator<<(std::ostream& os, const TensorPath& tensorPath) {
  return os << tensorPath.getTensor().getName() << ": "
            << "->" << util::join(tensorPath.getPath(), "->");
}


// class IterationSchedule
struct IterationSchedule::Content {
  vector<vector<Var>> indexVariables;
  vector<TensorPath> tensorPaths;
};

IterationSchedule::IterationSchedule() {
}

map<Var,set<Var>> getNeighborMap(const vector<TensorPath>& tensorPaths) {
  map<Var,set<Var>> neighbors;
  for (auto& tensorPath : tensorPaths) {
    auto path = tensorPath.getPath();
    for (size_t i=1; i < path.size(); ++i) {
      neighbors[path[i-1]].insert(path[i]);
    }
  }
  return neighbors;
}

IterationSchedule IterationSchedule::make(const taco::Expr& expr) {
  // Retrieve the tensor paths formed by the tensor reads. These paths contain
  // the edges of an iteration graph.
  struct CollectTensorPaths : public ExprVisitor {
    vector<TensorPath> tensorPaths;
    void visit(const ReadNode* op) {
      tensorPaths.push_back(TensorPath(op->tensor, op->indexVars));
    }
  };
  CollectTensorPaths collect;
  expr.accept(&collect);
  auto tensorPaths = collect.tensorPaths;

  // Find the source index variables (no incoming edges in the iteration graph)
  set<Var> indexVars;
  set<Var> notSources;
  for (auto& tensorPath : tensorPaths) {
    auto path = tensorPath.getPath();
    for (auto it = path.begin(); it != path.end(); ++it) {
      indexVars.insert(*it);
    }
    for (auto it = path.begin()+1; it != path.end(); ++it) {
      notSources.insert(*it);
    }
  }
  set<Var> sources = indexVars;
  sources.erase(notSources.begin(), notSources.end());

  // Compute the level of each index variable in the iteration graph. An index
  // variable's level is it's distance from a source index variable
  map<Var,int> levels;
  int maxLevel = 0;
  queue<Var> varsToVisit;
  for (auto& source : sources) {
    levels[source] = 0;
    varsToVisit.push(source);
  }
  auto neighbors = getNeighborMap(tensorPaths);
  while (varsToVisit.size() != 0) {
    Var var = varsToVisit.front();
    varsToVisit.pop();

    for (auto& succ : neighbors[var]) {
      if (!util::contains(levels, succ)) {
        int succLevel = levels[var] + 1;
        levels[succ] = succLevel;
        varsToVisit.push(succ);
        maxLevel = max(maxLevel, succLevel);
      }
    }
  }

  // Arrange index variables in levels according to the bfs results
  vector<vector<Var>> indexVariables(maxLevel+1);
  for (auto& varLevel : levels) {
    indexVariables[varLevel.second].push_back(varLevel.first);
  }

  IterationSchedule schedule = IterationSchedule();
  schedule.content = make_shared<IterationSchedule::Content>();

  schedule.content->indexVariables = indexVariables;
  schedule.content->tensorPaths = tensorPaths;

  std::cout << expr << std::endl;
  std::cout << schedule << std::endl;

  return schedule;
}

const vector<vector<taco::Var>>& IterationSchedule::getIndexVariables() const {
  return content->indexVariables;
}

const vector<TensorPath>& IterationSchedule::getTensorPaths() const {
  return content->tensorPaths;
}

std::ostream& operator<<(std::ostream& os, const IterationSchedule& schedule) {
  std::cout << "Index variables: " << std::endl;
  for (auto& level : schedule.getIndexVariables()) {
    std::cout << "  " << util::join(level) << std::endl;
  }
  std::cout << "Tensor paths:" << std::endl;
  for (auto& tensorPath : schedule.getTensorPaths()) {
    std::cout << "  " << tensorPath << std::endl;
  }
  return os;
}

}}
