#include "iteration_schedule_forest.h"

#include <iostream>
#include <set>
#include <queue>

#include "taco/util/error.h"
#include "taco/util/strings.h"
#include "taco/util/collections.h"

using namespace std;

namespace taco {
namespace lower {

/// Maps each index variable to its successors and predecessors through a path.
static tuple<set<Var>, set<Var>, map<Var, set<Var>>, map<Var,set<Var>>>
getGraph(const vector<TensorPath>& tensorPaths) {
  set<Var> vertices;
  set<Var> notSources;
  for (auto& tensorPath : tensorPaths) {
    auto steps = tensorPath.getVariables();
      for (auto it = steps.begin(); it != steps.end(); ++it) {
        vertices.insert(*it);
      }
    if (steps.size() > 0) {
      for (auto it = steps.begin()+1; it != steps.end(); ++it) {
        notSources.insert(*it);
      }
    }
  }
  set<Var> sources = vertices;
  for (auto& notSource : notSources) {
    sources.erase(notSource);
  }

  map<Var,set<Var>> successors;
  map<Var,set<Var>> predecessors;

  for (auto& var : vertices) {
    successors.insert({var, set<Var>()});
    predecessors.insert({var, set<Var>()});
  }

  // Traverse paths to insert successors and predecessors
  for (auto& tensorPath : tensorPaths) {
    auto path = tensorPath.getVariables();
    for (size_t i=1; i < path.size(); ++i) {
      successors.at(path[i-1]).insert(path[i]);
      predecessors.at(path[i]).insert(path[i-1]);
    }
  }

  return tuple<set<Var>, set<Var>, map<Var, set<Var>>, map<Var,set<Var>>>
		  {vertices, sources, successors, predecessors};
}

IterationScheduleForest::IterationScheduleForest(const vector<TensorPath>& paths) {
  // Construt a directed graph from the tensor paths
  set<Var> vertices;
  set<Var> sources;
  map<Var,set<Var>> successors;
  map<Var,set<Var>> predecessors;
  tie(vertices,sources,successors,predecessors) = getGraph(paths);

  // The sources of the path graph are the roots of the schedule forest
  roots.insert(roots.end(), sources.begin(), sources.end());

  // Compute the level of each index variable in the iteration graph. An index
  // variable's level is it's distance from a source index variable
  map<Var,int> levels;
  int maxLevel = 0;
  queue<Var> varsToVisit;
  for (auto& source : sources) {
    levels[source] = 0;
    varsToVisit.push(source);
  }
  while (varsToVisit.size() != 0) {
    Var var = varsToVisit.front();
    varsToVisit.pop();

    for (auto& successor : successors[var]) {
      int succLevel = levels[var] + 1;
      levels[successor] = succLevel;
      varsToVisit.push(successor);
      maxLevel = max(maxLevel, succLevel);
    }
  }
  taco_iassert(levels.size() == vertices.size());

  /// Initialize children vectors for all children
  for (auto& var : vertices) {
    children.insert({var, vector<Var>()});
  }

  // Construct the forest from the graph. The algorithm we use is:
  // - Use a BFS search to label all nodes with their level (above)
  // - For each node in reverse BFS orders
  //   - Make the predecessor with the highest level the parent
  //   - Make the parent a successor of other predecessors
  vector<pair<int,Var>> levelOrderedVars;
  for (auto& varLevel : levels) {
    levelOrderedVars.push_back({varLevel.second, varLevel.first});
  }
  // Sort in order of later level to former level
  std::sort(levelOrderedVars.begin(), levelOrderedVars.end(),
            [](pair<int,Var> a, pair<int,Var> b) {
              return b.first < a.first;
            });
  for (auto& levelVar : levelOrderedVars) {
    Var var = levelVar.second;

    auto& preds = predecessors.at(var);
    if (preds.size() > 0) {
      // Make the highest level predecessor the parent
      Var parent;
      int parentLevel = -1;
      for (auto& predecessor : preds) {
        int predecessorLevel = levels.at(predecessor);
        if (predecessorLevel > parentLevel) {
          parent      = predecessor;
          parentLevel = predecessorLevel;
        }
      }
      children.at(parent).push_back(var);
      parents.insert({var, parent});
      for (auto& predecessor : preds) {
        if (predecessor != parent) {
          // Make this predecessor a predecessor of parent so that it will
          // appear higher in the tree
          predecessors.at(parent).insert({predecessor});
          successors.at(predecessor).insert({parent});
        }
      }
    }
  }
}

bool IterationScheduleForest::hasParent(const Var& var) const {
  return util::contains(parents, var);
}

const Var& IterationScheduleForest::getParent(const Var& var) const {
  taco_iassert(hasParent(var)) << "Attempting to get the parent of " << var  <<
                             " which has no no parent";
  return parents.at(var);
}

const std::vector<Var>&
IterationScheduleForest::getChildren(const Var& var) const {
  taco_iassert(util::contains(children,var)) << var << " does not have any children";
  return children.at(var);
}

std::vector<Var> IterationScheduleForest::getNodes() const {
  std::vector<Var> nodes;
  for (auto& var : children) {
    nodes.push_back(var.first);
  }
  return nodes;
}

std::ostream& operator<<(std::ostream& os,
                         const IterationScheduleForest& forest) {
  os << "roots: " << util::join(forest.getRoots()) << std::endl;
  auto it = forest.children.begin();
  auto end = forest.children.end();
  if (it != end) {
    if (it->second.size() != 0) {
      os << it->first << " -> " << util::join(it->second);
    }
    it++;
  }
  while (it != end) {
    if (it->second.size() != 0) {
      os << std::endl << it->first << " -> " << util::join(it->second);
    }
    it++;
  }
  return os;
}

}}
