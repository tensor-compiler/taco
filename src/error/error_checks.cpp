#include "error_checks.h"

#include <map>
#include <set>
#include <stack>
#include <functional>

#include "taco/tensor.h"
#include "taco/expr.h"
#include "taco/expr_nodes/expr_nodes.h"
#include "taco/util/strings.h"
#include "taco/util/collections.h"

using namespace std;
using namespace taco::expr_nodes;

namespace taco {
namespace error {

static vector<const ReadNode*> getReadNodes(const IndexExpr& expr) {
  vector<const ReadNode*> readNodes;
  match(expr,
    std::function<void(const ReadNode*)>([&](const ReadNode* op) {
      readNodes.push_back(op);
    })
  );
  return readNodes;
}

bool dimensionsTypecheck(const std::vector<IndexVar>& resultVars,
                         const IndexExpr& expr,
                         const std::vector<int>& dimensions) {

  std::map<IndexVar,int> varSizes;
  for (size_t i = 0; i < resultVars.size(); i++) {
    IndexVar var = resultVars[i];
    int dimension = dimensions[i];
    if (util::contains(varSizes, var) && varSizes.at(var) != dimension) {
      return false;
    }
    else {
      varSizes.insert({var, dimension});
    }
  }

  vector<const ReadNode*> readNodes = getReadNodes(expr);
  for (auto& readNode : readNodes) {
    for (size_t i = 0; i < readNode->indexVars.size(); i++) {
      IndexVar var = readNode->indexVars[i];
      int dimension = readNode->tensor.getDimension(i);
      if (util::contains(varSizes, var) && varSizes.at(var) != dimension) {
        return false;
      }
      else {
        varSizes.insert({var, dimension});
      }
    }
  }

  return true;
}

static string addDimensionError(const IndexVar& var, int dim1, int dim2) {
  return "Index variable " + util::toString(var) + " is used to index "
         "dimensions of different sizes (" + util::toString(dim1) +
         " and " + util::toString(dim2) + ").";
}

std::string dimensionTypecheckErrors(const std::vector<IndexVar>& resultVars,
                                     const IndexExpr& expr,
                                     const std::vector<int>& dimensions) {
  vector<string> errors;

  std::map<IndexVar,int> varSizes;
  for (size_t i = 0; i < resultVars.size(); i++) {
    IndexVar var = resultVars[i];
    int dimension = dimensions[i];
    if (util::contains(varSizes, var) && varSizes.at(var) != dimension) {
      errors.push_back(addDimensionError(var, varSizes.at(var), dimension));
    }
    else {
      varSizes.insert({var, dimension});
    }
  }

  vector<const ReadNode*> readNodes = getReadNodes(expr);
  for (auto& readNode : readNodes) {
    for (size_t i = 0; i < readNode->indexVars.size(); i++) {
      IndexVar var = readNode->indexVars[i];
      int dimension = readNode->tensor.getDimension(i);
      if (util::contains(varSizes, var) && varSizes.at(var) != dimension) {
        errors.push_back(addDimensionError(var, varSizes.at(var), dimension));
      }
      else {
        varSizes.insert({var, dimension});
      }
    }
  }

  return util::join(errors, " ");
}

static void addEdges(vector<IndexVar> indexVars, vector<int> dimOrder,
                     map<IndexVar,set<IndexVar>>* successors) {
  if (indexVars.size() == 0) {
    return;
  }

  for (size_t i = 0; i < dimOrder.size()-1; i++) {
    IndexVar var = indexVars[dimOrder[i]];
    IndexVar succ = indexVars[dimOrder[i+1]];
    if (!util::contains(*successors, var)) {
      successors->insert({var, set<IndexVar>()});
    }
    successors->at(var).insert(succ);
  }
  IndexVar var = indexVars[dimOrder[dimOrder.size()-1]];
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
  // tensor access expressions, where edges follow the tensor format order,
  // contains a cycle.
  map<IndexVar,set<IndexVar>> successors;

  addEdges(resultVars, resultFormat.getModeOrder(), &successors);
  match(expr,
    std::function<void(const ReadNode*)>([&successors](const ReadNode* op) {
      addEdges(op->indexVars, op->tensor.getFormat().getModeOrder(),
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

bool containsDistribution(const std::vector<IndexVar>& resultVars,
                          const IndexExpr& expr) {
  // We don't yet support distributing tensors. That is, every free variable
  // must be used on the right-hand-side.
  set<IndexVar> rhsVars;
  using namespace expr_nodes;
  match(expr,
    function<void(const ReadNode*)>([&](const ReadNode* op) {
      for (auto& var : op->indexVars) {
        rhsVars.insert(var);
      }
    })
  );
  for (auto& lhsVar : resultVars) {
    if (!util::contains(rhsVars, lhsVar)) {
      return true;
    }
  }
  return false;
}

}}
