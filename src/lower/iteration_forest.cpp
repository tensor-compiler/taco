#include "iteration_forest.h"

#include <algorithm>
#include <iostream>
#include <set>
#include <queue>

#include "taco/error.h"
#include "taco/util/strings.h"
#include "taco/util/collections.h"

using namespace std;

namespace taco {

/// Maps each index variable to its successors and predecessors through a path.
static tuple<vector<IndexVar>,
             set<IndexVar>,
             map<IndexVar,set<IndexVar>>,
             map<IndexVar,set<IndexVar>>>
getGraph(const vector<TensorPath>& tensorPaths) {
  vector<IndexVar> vertices;
  set<IndexVar> notSources;
  for (auto& tensorPath : tensorPaths) {
    auto steps = tensorPath.getVariables();
      for (auto it = steps.begin(); it != steps.end(); ++it) {
        if (!util::contains(vertices, *it)) {
          vertices.push_back(*it);
        }
      }
    if (steps.size() > 0) {
      for (auto it = steps.begin()+1; it != steps.end(); ++it) {
        notSources.insert(*it);
      }
    }
  }
  set<IndexVar> sources(vertices.begin(), vertices.end());
  for (auto& notSource : notSources) {
    sources.erase(notSource);
  }

  map<IndexVar,set<IndexVar>> successors;
  map<IndexVar,set<IndexVar>> predecessors;

  for (auto& var : vertices) {
    successors.insert({var, set<IndexVar>()});
    predecessors.insert({var, set<IndexVar>()});
  }

  // Traverse paths to insert successors and predecessors
  for (auto& tensorPath : tensorPaths) {
    auto path = tensorPath.getVariables();
    for (size_t i=1; i < path.size(); ++i) {
      successors.at(path[i-1]).insert(path[i]);
      predecessors.at(path[i]).insert(path[i-1]);
    }
  }

  return tuple<vector<IndexVar>,
               set<IndexVar>,
               map<IndexVar,set<IndexVar>>,
               map<IndexVar,set<IndexVar>>>
		  {vertices, sources, successors, predecessors};
}

IterationForest::IterationForest(const vector<TensorPath>& paths) {
  // Construt a directed graph from the tensor paths
  vector<IndexVar> vertices;
  set<IndexVar> sources;
  map<IndexVar,set<IndexVar>> successors;
  map<IndexVar,set<IndexVar>> predecessors;
  tie(vertices,sources,successors,predecessors) = getGraph(paths);

  // The sources of the path graph are the roots of the iteration forest
  roots.insert(roots.end(), sources.begin(), sources.end());

  // Compute the level of each index variable in the iteration graph. An index
  // variable's level is it's distance from a source index variable
  map<IndexVar,int> levels;
  int maxLevel = 0;
  queue<IndexVar> varsToVisit;
  for (auto& source : sources) {
    levels[source] = 0;
    varsToVisit.push(source);
  }
  while (varsToVisit.size() != 0) {
    IndexVar var = varsToVisit.front();
    varsToVisit.pop();

    for (auto& successor : successors[var]) {
      int succLevel = levels[var] + 1;
      levels[successor] = succLevel;
      varsToVisit.push(successor);
      maxLevel = std::max(maxLevel, succLevel);
    }
  }
  taco_iassert(levels.size() == vertices.size());

  /// Initialize children vectors for all vertices
  for (auto& var : vertices) {
    children.insert({var, vector<IndexVar>()});
  }

  // Construct the forest from the graph. The algorithm we use is:
  // - Use a BFS search to label all nodes with their level (above)
  // - For each node in reverse BFS orders
  //   - Make the predecessor with the highest level the parent
  //   - Make the parent a successor of other predecessors
  vector<pair<int,IndexVar>> levelOrderedVars;
  for (auto& varLevel : levels) {
    levelOrderedVars.push_back({varLevel.second, varLevel.first});
  }
  // Sort in order of later level to former level
  sort(levelOrderedVars.begin(), levelOrderedVars.end(),
       [](pair<int,IndexVar> a, pair<int,IndexVar> b) {
         return b.first < a.first;
       });
  for (auto& levelVar : levelOrderedVars) {
    IndexVar indexVar = levelVar.second;

    auto& preds = predecessors.at(indexVar);
    if (preds.size() > 0) {
      // Make the highest level predecessor the parent
      IndexVar parent;
      int parentLevel = -1;
      for (auto& predecessor : preds) {
        int predecessorLevel = levels.at(predecessor);
        if (predecessorLevel > parentLevel) {
          parent      = predecessor;
          parentLevel = predecessorLevel;
        }
      }
      children.at(parent).push_back(indexVar);
      parents.insert({indexVar, parent});
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

  // Sort children vectors in order of total vertex ordering, to get
  // deterministic loop sequencing
  map<IndexVar,int> vertexPositions;
  for (size_t i = 0; i < vertices.size(); i++) {
    vertexPositions.insert({vertices[i], i});
  }
  for (auto& vertex : vertices) {
    vector<pair<int, IndexVar>> childPositions;
    for (auto& child : children.at(vertex)) {
      childPositions.push_back({vertexPositions.at(child), child});
    }
    sort(childPositions.begin(), childPositions.end(),
         [](pair<int,IndexVar> a, pair<int,IndexVar> b) {
           return b.first > a.first;
         });
    vector<IndexVar> orderedChildren;
    for (auto& childPos : childPositions) {
      orderedChildren.push_back(childPos.second);
    }
    children.at(vertex) = orderedChildren;
  }
}

bool IterationForest::hasParent(const IndexVar& var) const {
  return util::contains(parents, var);
}

const IndexVar& IterationForest::getParent(const IndexVar& var) const {
  taco_iassert(hasParent(var)) <<
      "Attempting to get the parent of " << var  << " which has no no parent";
  return parents.at(var);
}

const std::vector<IndexVar>&
IterationForest::getChildren(const IndexVar& var) const {
  taco_iassert(util::contains(children,var)) <<
      var << " does not have any children";
  return children.at(var);
}

std::vector<IndexVar> IterationForest::getNodes() const {
  std::vector<IndexVar> nodes;
  for (auto& var : children) {
    nodes.push_back(var.first);
  }
  return nodes;
}

std::ostream& operator<<(std::ostream& os, const IterationForest& forest) {
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

}
