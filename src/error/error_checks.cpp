#include "error_checks.h"

#include <map>
#include <set>
#include <stack>
#include <functional>

#include "taco/type.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/util/strings.h"
#include "taco/util/collections.h"

using namespace std;

namespace taco {
namespace error {

static vector<const AccessNode*> getAccessNodes(const IndexExpr& expr) {
  vector<const AccessNode*> readNodes;
  match(expr,
    std::function<void(const AccessNode*)>([&](const AccessNode* op) {
      readNodes.push_back(op);
    })
  );
  return readNodes;
}

bool dimensionsTypecheck(const std::vector<IndexVar>& resultVars,
                         const IndexExpr& expr,
                         const Shape& shape) {

  std::map<IndexVar,Dimension> indexVarDims;
  for (size_t mode = 0; mode < resultVars.size(); mode++) {
    IndexVar var = resultVars[mode];
    auto dimension = shape.getDimension(mode);
    if (util::contains(indexVarDims,var) && indexVarDims.at(var) != dimension) {
      return false;
    }
    else {
      indexVarDims.insert({var, dimension});
    }
  }

  vector<const AccessNode*> readNodes = getAccessNodes(expr);
  for (auto& readNode : readNodes) {
    for (size_t mode = 0; mode < readNode->indexVars.size(); mode++) {
      IndexVar var = readNode->indexVars[mode];
      Dimension dimension =
          readNode->tensorVar.getType().getShape().getDimension(mode);
      if (util::contains(indexVarDims,var) &&
          indexVarDims.at(var) != dimension) {
        return false;
      }
      else {
        indexVarDims.insert({var, dimension});
      }
    }
  }

  return true;
}

static string addDimensionError(const IndexVar& var,
                                Dimension dimension1, Dimension dimension2) {
  return "Index variable " + util::toString(var) + " is used to index "
         "modes of different dimensions (" + util::toString(dimension1) +
         " and " + util::toString(dimension2) + ").";
}

std::string dimensionTypecheckErrors(const std::vector<IndexVar>& resultVars,
                                     const IndexExpr& expr,
                                     const Shape& shape) {
  vector<string> errors;

  std::map<IndexVar,Dimension> indexVarDims;
  for (size_t mode = 0; mode < resultVars.size(); mode++) {
    IndexVar var = resultVars[mode];
    auto dimension = shape.getDimension(mode);
    if (util::contains(indexVarDims,var) && indexVarDims.at(var) != dimension) {
      errors.push_back(addDimensionError(var, indexVarDims.at(var), dimension));
    }
    else {
      indexVarDims.insert({var, dimension});
    }
  }

  vector<const AccessNode*> readNodes = getAccessNodes(expr);
  for (auto& readNode : readNodes) {
    for (size_t mode = 0; mode < readNode->indexVars.size(); mode++) {
      IndexVar var = readNode->indexVars[mode];
      Dimension dimension =
          readNode->tensorVar.getType().getShape().getDimension(mode);
      if (util::contains(indexVarDims,var) &&
          indexVarDims.at(var) != dimension) {
        errors.push_back(addDimensionError(var, indexVarDims.at(var),
                                           dimension));
      }
      else {
        indexVarDims.insert({var, dimension});
      }
    }
  }

  return util::join(errors, " ");
}

static void addEdges(vector<IndexVar> indexVars, vector<int> modeOrdering,
                     map<IndexVar,set<IndexVar>>* successors) {
  if (indexVars.size() == 0) {
    return;
  }

  for (size_t i = 0; i < modeOrdering.size()-1; i++) {
    IndexVar var = indexVars[modeOrdering[i]];
    IndexVar succ = indexVars[modeOrdering[i+1]];
    if (!util::contains(*successors, var)) {
      successors->insert({var, set<IndexVar>()});
    }
    successors->at(var).insert(succ);
  }
  IndexVar var = indexVars[modeOrdering[modeOrdering.size()-1]];
  if (!util::contains(*successors, var)) {
    successors->insert({var, set<IndexVar>()});
  }
}

static bool hasCycle(const IndexVar& var,
                     const map<IndexVar,set<IndexVar>>& successors,
                     set<IndexVar>* visited, set<IndexVar>* marked) {
  if (!util::contains(*visited, var)) {
    visited->insert(var);
    marked->insert(var);

    for (auto& succ : successors.at(var)) {
      if (!util::contains(*visited, succ) &&
          hasCycle(succ, successors, visited, marked)) {
        return true;
      }
      else if (util::contains(*marked, succ)) {
        return true;
      }
    }
  }
  marked->erase(var);
  return false;
}

bool containsTranspose(const Format& resultFormat,
                       const std::vector<IndexVar>& resultVars,
                       const IndexExpr& expr) {
  // An index expression contains a transposition if a graph constructed from
  // tensor access expressions, where edges follow the tensor mode ordering,
  // contains a cycle.
  map<IndexVar,set<IndexVar>> successors;

  addEdges(resultVars, resultFormat.getModeOrdering(), &successors);
  match(expr,
    std::function<void(const AccessNode*)>([&successors](const AccessNode* op) {
      addEdges(op->indexVars, op->tensorVar.getFormat().getModeOrdering(),
               &successors);
    })
  );

  set<IndexVar> visited;
  set<IndexVar> marked;
  for (auto& indexVar : successors) {
    if (hasCycle(indexVar.first, successors, &visited, &marked)) {
      return true;
    }
  }
  return false;
}

}}
