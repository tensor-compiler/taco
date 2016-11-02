#include "iteration_schedule.h"

#include <set>
#include <vector>
#include <queue>

#include "var.h"
#include "expr_nodes.h"
#include "expr_visitor.h"
#include "internal_tensor.h"
#include "tensor_path.h"
#include "merge_rule.h"

#include "util/strings.h"
#include "util/collections.h"

using namespace std;

namespace taco {
namespace lower {

// class IterationSchedule
struct IterationSchedule::Content {
  Content(internal::Tensor tensor, vector<vector<Var>> indexVariables,
          TensorPath resultTensorPath, vector<TensorPath> tensorPaths,
          map<Var,MergeRule> mergeRules,
          map<Expr,TensorPath> mapReadNodesToPaths)
      : tensor(tensor), indexVariables(indexVariables),
        resultTensorPath(resultTensorPath), tensorPaths(tensorPaths),
        mergeRules(mergeRules), mapReadNodesToPaths(mapReadNodesToPaths) {}

  internal::Tensor     tensor;
  vector<vector<Var>>  indexVariables;

  TensorPath           resultTensorPath;
  vector<TensorPath>   tensorPaths;

  map<Var,MergeRule>   mergeRules;
  map<Expr,TensorPath> mapReadNodesToPaths;
};

IterationSchedule::IterationSchedule() {
}

map<Var,set<Var>> getNeighborMap(const vector<TensorPath>& tensorPaths);
map<Var,set<Var>> getNeighborMap(const vector<TensorPath>& tensorPaths) {
  map<Var,set<Var>> neighbors;
  for (auto& tensorPath : tensorPaths) {
    auto path = tensorPath.getVariables();
    for (size_t i=1; i < path.size(); ++i) {
      neighbors[path[i-1]].insert(path[i]);
    }
  }
  return neighbors;
}

/// Arrange the index variables in levels according to a bfs traversal of the
/// graph created by the given tensor paths.
static vector<vector<Var>>
arrangeIndexVariables(const vector<TensorPath>& tensorPaths) {
  // Find the source index variables (no incoming edges in the iteration graph)
  set<Var> indexVars;
  set<Var> notSources;
  for (auto& tensorPath : tensorPaths) {
    auto path = tensorPath.getVariables();
    for (auto it = path.begin(); it != path.end(); ++it) {
      indexVars.insert(*it);
    }
    for (auto it = path.begin()+1; it != path.end(); ++it) {
      notSources.insert(*it);
    }
  }
  set<Var> sources = indexVars;
  for (auto& notSource : notSources) {
    sources.erase(notSource);
  }

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

  return indexVariables;
}

static
map<Var,MergeRule> createMergeRules(const internal::Tensor& tensor,
                                    vector<vector<Var>> indexVariables,
                                    map<Expr,TensorPath> tensorPaths,
                                    const TensorPath& resultTensorPath) {
  map<Var,MergeRule> mergeRules;
  for (auto& vars : indexVariables) {
    for (auto& var : vars) {
      mergeRules.insert({var, MergeRule::make(tensor, var, tensorPaths,
                                              resultTensorPath)});
    }
  }
  return mergeRules;
}

IterationSchedule IterationSchedule::make(const internal::Tensor& tensor) {
  Expr expr = tensor.getExpr();

  // Create the tensor path formed by the result.
  TensorPath resultTensorPath = TensorPath(tensor, tensor.getIndexVars());

  // Create the paths formed by tensor reads in the given expression.
  struct CollectTensorPaths : public internal::ExprVisitor {
    using ExprVisitor::visit;
    vector<TensorPath> tensorPaths;
    map<Expr,TensorPath> mapReadNodesToPaths;
    void visit(const internal::Read* op) {
      auto tensorPath = TensorPath(op->tensor, op->indexVars);
      mapReadNodesToPaths.insert({op, tensorPath});
      tensorPaths.push_back(tensorPath);
    }
  };
  CollectTensorPaths collect;
  expr.accept(&collect);
  vector<TensorPath> tensorPaths = collect.tensorPaths;
  map<Expr,TensorPath> mapReadNodesToPaths = collect.mapReadNodesToPaths;

  // Arrange index variables in levels. Each level will result in one loop nest
  // and the variables inside a level result in a loop sequence.
  vector<vector<Var>> indexVariables = arrangeIndexVariables(tensorPaths);

  // Create merge rules that describe how to merge the tensor paths incomming
  // on each index variable.

  map<Var,MergeRule> mergeRules = createMergeRules(tensor, indexVariables,
                                                   mapReadNodesToPaths,
                                                   resultTensorPath);

  // Create the iteration schedule
  IterationSchedule schedule = IterationSchedule();
  schedule.content =
      make_shared<IterationSchedule::Content>(tensor,
                                              indexVariables,
                                              resultTensorPath,
                                              tensorPaths,
                                              mergeRules,
                                              mapReadNodesToPaths);
  return schedule;
}

const internal::Tensor& IterationSchedule::getTensor() const {
  return content->tensor;
}

size_t IterationSchedule::numLayers() const {
  return getIndexVariables().size();
}

const vector<vector<taco::Var>>& IterationSchedule::getIndexVariables() const {
  return content->indexVariables;
}

const MergeRule& IterationSchedule::getMergeRule(const taco::Var& var) const {
  iassert(util::contains(content->mergeRules, var))
      << "No merge rule for variable " << var;
  return content->mergeRules.at(var);
}

const vector<TensorPath>& IterationSchedule::getTensorPaths() const {
  return content->tensorPaths;
}

const TensorPath&
IterationSchedule::getTensorPath(const taco::Expr& operand) const {
  iassert(util::contains(content->mapReadNodesToPaths, operand));
  return content->mapReadNodesToPaths.at(operand);
}

const TensorPath& IterationSchedule::getResultTensorPath() const {
  return content->resultTensorPath;
}

std::ostream& operator<<(std::ostream& os, const IterationSchedule& schedule) {
  os << "Index variables: " << std::endl;
  for (auto& level : schedule.getIndexVariables()) {
    os << "  " << util::join(level) << std::endl;
  }
  os << "Merge rules:" << std::endl;
  for (auto& level : schedule.getIndexVariables()) {
    for (auto& var : level) {
      os << "  " << var << ": " << schedule.getMergeRule(var) << std::endl;
    }
  }
  os << "Result tensor path" << std::endl;
  os << "  " << schedule.getResultTensorPath() << std::endl;
  os << "Tensor paths:" << std::endl;
  for (auto& tensorPath : schedule.getTensorPaths()) {
    os << "  " << tensorPath << std::endl;
  }
  return os;
}

}}
