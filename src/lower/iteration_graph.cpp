#include "iteration_graph.h"

#include <set>
#include <vector>
#include <queue>

#include "taco/tensor.h"
#include "taco/lower/schedule.h"
#include "taco/expr_nodes/expr_nodes.h"
#include "taco/expr_nodes/expr_visitor.h"
#include "iteration_forest.h"
#include "tensor_path.h"
#include "taco/util/strings.h"
#include "taco/util/collections.h"

using namespace std;

namespace taco {
namespace lower {

// class IterationGraph
struct IterationGraph::Content {
  Content(IterationForest iterationForest, const vector<IndexVar>& freeVars,
          TensorPath resultTensorPath, vector<TensorPath> tensorPaths,
          map<IndexExpr,TensorPath> mapAccessNodesToPaths)
      : iterationForest(iterationForest),
        freeVars(freeVars.begin(), freeVars.end()),
        resultTensorPath(resultTensorPath),
        tensorPaths(tensorPaths),
        mapAccessNodesToPaths(mapAccessNodesToPaths) {}
  IterationForest           iterationForest;
  set<IndexVar>             freeVars;

  TensorPath                resultTensorPath;
  vector<TensorPath>        tensorPaths;
  map<IndexExpr,TensorPath> mapAccessNodesToPaths;
};

IterationGraph::IterationGraph() {
}

IterationGraph IterationGraph::make(const TensorBase& tensor,
                                    const Schedule& schedule) {
  IndexExpr expr = tensor.getExpr();

  vector<TensorPath> tensorPaths;

  // Create the tensor path formed by the result.
  vector<IndexVar> resultIndexVars;
  for (size_t i = 0; i < tensor.getOrder(); ++i) {
    size_t idx = tensor.getFormat().getModeOrdering()[i];
    resultIndexVars.push_back(tensor.getIndexVars()[idx]);
  }
  TensorPath resultTensorPath = TensorPath(tensor, resultIndexVars);

  // Create the paths formed by tensor reads in the given expression.
  struct CollectTensorPaths : public expr_nodes::ExprVisitor {
    using ExprVisitor::visit;
    vector<TensorPath> tensorPaths;
    map<IndexExpr,TensorPath> mapAccessNodesToPaths;
    void visit(const expr_nodes::AccessNode* op) {
      taco_iassert(op->tensor.getOrder() == op->indexVars.size()) <<
          "Tensor access " << IndexExpr(op) << " but tensor format only has " <<
          op->tensor.getOrder() << " modes.";
      Format format = op->tensor.getFormat();

      // copy index variables to path
      vector<IndexVar> path(op->indexVars.size());
      for (size_t i=0; i < op->indexVars.size(); ++i) {
        path[i] = op->indexVars[format.getModeOrdering()[i]];
      }

      auto tensorPath = TensorPath(op->tensor, path);
      mapAccessNodesToPaths.insert({op, tensorPath});
      tensorPaths.push_back(tensorPath);
    }
  };
  CollectTensorPaths collect;
  expr.accept(&collect);
  util::append(tensorPaths, collect.tensorPaths);
  map<IndexExpr,TensorPath> mapAccessNodesToPaths = collect.mapAccessNodesToPaths;

  // Construct a forest decomposition from the tensor path graph
  IterationForest forest =
      IterationForest(util::combine({resultTensorPath},tensorPaths));

  // Create the iteration graph
  IterationGraph iterationGraph = IterationGraph();
  iterationGraph.content =
      make_shared<IterationGraph::Content>(forest, tensor.getIndexVars(),
                                           resultTensorPath, tensorPaths,
                                           mapAccessNodesToPaths);
  return iterationGraph;
}

const std::vector<IndexVar>& IterationGraph::getRoots() const {
  return content->iterationForest.getRoots();
}

const IndexVar& IterationGraph::getParent(const IndexVar& var) const {
  return content->iterationForest.getParent(var);
}

const std::vector<IndexVar>&
IterationGraph::getChildren(const IndexVar& var) const {
  return content->iterationForest.getChildren(var);
}

vector<IndexVar> IterationGraph::getAncestors(const IndexVar& var) const {
  std::vector<IndexVar> ancestors;
  ancestors.push_back(var);
  IndexVar parent = var;
  while (content->iterationForest.hasParent(parent)) {
    parent = content->iterationForest.getParent(parent);
    ancestors.push_back(parent);
  }
  return ancestors;
}

vector<IndexVar> IterationGraph::getDescendants(const IndexVar& var) const{
  vector<IndexVar> descendants;
  descendants.push_back(var);
  for (auto& child : getChildren(var)) {
    util::append(descendants, getDescendants(child));
  }
  return descendants;
}

bool IterationGraph::isLastFreeVariable(const IndexVar& var) const {
  return isFree(var) && !hasFreeVariableDescendant(var);
}

bool IterationGraph::hasFreeVariableDescendant(const IndexVar& var) const {
  // Traverse the iteration forest subtree of var to determine whether it has
  // any free variable descendants
  auto children = content->iterationForest.getChildren(var);
  for (auto& child : children) {
    if (isFree(child)) {
      return true;
    }
    // Child is not free; check if it a free descendent
    if (hasFreeVariableDescendant(child)) {
      return true;
    }
  }
  return false;
}

bool IterationGraph::hasReductionVariableAncestor(const IndexVar& var) const {
  if (isReduction(var)) {
    return true;
  }

  IndexVar parent = var;
  while (content->iterationForest.hasParent(parent)) {
    parent = content->iterationForest.getParent(parent);
    if (isReduction(parent)) {
      return true;
    }
  }
  return false;
}

const vector<TensorPath>& IterationGraph::getTensorPaths() const {
  return content->tensorPaths;
}

const TensorPath&
IterationGraph::getTensorPath(const IndexExpr& operand) const {
  taco_iassert(util::contains(content->mapAccessNodesToPaths, operand));
  return content->mapAccessNodesToPaths.at(operand);
}

const TensorPath& IterationGraph::getResultTensorPath() const {
  return content->resultTensorPath;
}

IndexVarType IterationGraph::getIndexVarType(const IndexVar& var) const {
  return (util::contains(content->freeVars, var))
      ? IndexVarType::Free : IndexVarType::Sum;
}

bool IterationGraph::isFree(const IndexVar& var) const {
  return getIndexVarType(var) == IndexVarType::Free;
}

bool IterationGraph::isReduction(const IndexVar& var) const {
  return !isFree(var);
}

std::ostream& operator<<(std::ostream& os, const IterationGraph& graph) {
  os << "Index Variable Forest" << std::endl;
  os << graph.content->iterationForest << std::endl;
  os << "Result tensor path" << std::endl;
  os << "  " << graph.getResultTensorPath() << std::endl;
  os << "Tensor paths:" << std::endl;
  for (auto& tensorPath : graph.getTensorPaths()) {
    os << "  " << tensorPath << std::endl;
  }
  return os;
}

}}
