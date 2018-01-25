#include "iteration_graph.h"

#include <set>
#include <vector>
#include <queue>
#include <functional>

#include "taco/expr/expr.h"
#include "taco/expr/expr_nodes.h"
#include "taco/expr/expr_visitor.h"
#include "taco/expr/schedule.h"
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
          map<IndexExpr,TensorPath> mapAccessNodesToPaths, IndexExpr expr)
      : iterationForest(iterationForest),
        freeVars(freeVars.begin(), freeVars.end()),
        resultTensorPath(resultTensorPath),
        tensorPaths(tensorPaths),
        accessNodesToPaths(mapAccessNodesToPaths),
        expr(expr) {
  }
  IterationForest           iterationForest;
  set<IndexVar>             freeVars;

  TensorPath                resultTensorPath;
  vector<TensorPath>        tensorPaths;

  vector<TensorVar>         workspaces;

  map<IndexExpr,TensorPath> accessNodesToPaths;

  // TODO: This must be replaced by a map that maps each index variable to a
  //       (potentially rewritten) index expression.
  IndexExpr                 expr;
};

IterationGraph::IterationGraph() {
}

IterationGraph IterationGraph::make(const TensorVar& tensor) {
  IndexExpr expr = tensor.getIndexExpr();

  vector<TensorPath> tensorPaths;
  vector<TensorVar> workspaces;
  map<IndexExpr,TensorPath> accessNodesToPaths;

  map<IndexVar,Dimension> indexVarRanges = getIndexVarRanges(tensor);

  map<IndexVar,IndexVar> oldToSplitVar;  // remap split index variables
  for (auto& indexVarRange : indexVarRanges) {
    auto indexVar = indexVarRange.first;
    oldToSplitVar.insert({indexVar, indexVar});
  }

  match(expr,
    function<void(const MulNode*,Matcher*)>([&](const MulNode* op,Matcher* ctx){
      // Remap split index variables (old) to the left index variables.
      for (auto& osplit : op->getOperatorSplits()) {
        oldToSplitVar[osplit.getOld()] = osplit.getLeft();

        // Add result workspace
        Type type(Float(), {indexVarRanges.at(osplit.getOld())});
        TensorVar workspace("w", type, Dense);
        workspaces.push_back(workspace);

        // Add path to the left variable to store to workspace
        TensorPath workspaceResultPath({osplit.getLeft()},
                                       Access(workspace, {osplit.getLeft()}));
        tensorPaths.push_back(workspaceResultPath);

        // Add path to the old variable to load from workspace
        TensorPath workspaceOperandPath({osplit.getOld()},
                                        Access(workspace, {osplit.getOld()}));
        tensorPaths.push_back(workspaceOperandPath);
      }

      ctx->match(op->a);

      // Clean up mapping
      for (auto& osplit : op->getOperatorSplits()) {
        oldToSplitVar[osplit.getOld()] = osplit.getOld();
      }

      ctx->match(op->b);
    }),

    function<void(const AccessNode*)>([&](const AccessNode* op) {
      auto type = op->tensorVar.getType();
      taco_iassert(type.getShape().getOrder() == op->indexVars.size()) <<
          "Tensor access " << IndexExpr(op) << " but tensor format only has " <<
          type.getShape().getOrder() << " modes.";
      Format format = op->tensorVar.getFormat();

      // copy index variables to path
      vector<IndexVar> path(op->indexVars.size());
      for (size_t i=0; i < op->indexVars.size(); ++i) {
        int ordering = op->tensorVar.getFormat().getModeOrdering()[i];
        path[i] = oldToSplitVar.at(op->indexVars[ordering]);
      }

      TensorPath tensorPath(path, op);
      accessNodesToPaths.insert({op, tensorPath});
      tensorPaths.push_back(tensorPath);
    })
  );

  vector<IndexVar> resultVars;
  for (size_t i = 0; i < tensor.getType().getShape().getOrder(); ++i) {
    size_t idx = tensor.getFormat().getModeOrdering()[i];
    resultVars.push_back(tensor.getFreeVars()[idx]);
  }
  TensorPath resultPath = TensorPath(resultVars,
                                     Access(tensor, tensor.getFreeVars()));

  // Construct a forest decomposition from the tensor path graph
  IterationForest forest =
      IterationForest(util::combine({resultPath}, tensorPaths));

  // Create the iteration graph
  IterationGraph iterationGraph = IterationGraph();
  iterationGraph.content =
      make_shared<IterationGraph::Content>(forest, tensor.getFreeVars(),
                                           resultPath, tensorPaths,
                                           accessNodesToPaths, expr);
  return iterationGraph;
}

const std::vector<IndexVar>& IterationGraph::getRoots() const {
  return content->iterationForest.getRoots();
}

const std::vector<IndexVar>&
IterationGraph::getChildren(const IndexVar& var) const {
  return content->iterationForest.getChildren(var);
}

const IndexVar& IterationGraph::getParent(const IndexVar& var) const {
  return content->iterationForest.getParent(var);
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

const vector<TensorPath>& IterationGraph::getTensorPaths() const {
  return content->tensorPaths;
}

const TensorPath&
IterationGraph::getTensorPath(const IndexExpr& operand) const {
  taco_iassert(util::contains(content->accessNodesToPaths, operand));
  return content->accessNodesToPaths.at(operand);
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

const IndexExpr& IterationGraph::getIndexExpr(const IndexVar&) const {
  return content->expr;
}

void IterationGraph::printAsDot(std::ostream& os) {
  os << "digraph {";
  os << "\n root [label=\"\" shape=none]";

  for (auto& path : getTensorPaths()) {
    string name = path.getAccess().getTensorVar().getName();
    auto& vars = path.getVariables();
    if (vars.size() > 0) {
      os << "\n root -> " << vars[0]
         << " [label=\"" << name << "\"]";
    }
  }

  auto& resultPath = getResultTensorPath();
  string resultName = resultPath.getAccess().getTensorVar().getName();
  auto& resultVars = resultPath.getVariables();
  if (resultVars.size() > 0) {
    os << "\n root -> " << resultVars[0]
       << " [style=dashed label=\"" << resultName << "\"]";
  }

  for (auto& path : getTensorPaths()) {
    string name = path.getAccess().getTensorVar().getName();
    auto& vars = path.getVariables();
    for (size_t i = 1; i < vars.size(); i++) {
      os << "\n " << vars[i-1] << " -> " << vars[i]
         << " [label=\"" << name << "\"]";
    }
  }

  for (size_t i = 1; i < resultVars.size(); i++) {
    os << "\n " << resultVars[i-1] << " -> " << resultVars[i]
       << " [style=dashed label=\"" << resultName << "\"]";
  }
  os << "\n}\n";
  os.flush();
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
