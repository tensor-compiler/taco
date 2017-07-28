#include "iteration_schedule.h"

#include <set>
#include <vector>
#include <queue>

#include "taco/tensor.h"
#include "taco/expr_nodes/expr_nodes.h"
#include "taco/expr_nodes/expr_visitor.h"
#include "iteration_schedule_forest.h"
#include "tensor_path.h"
#include "taco/util/strings.h"
#include "taco/util/collections.h"

using namespace std;

namespace taco {
namespace lower {

// class IterationSchedule
struct IterationSchedule::Content {
  Content(TensorBase tensor, IterationScheduleForest scheduleForest,
          TensorPath resultTensorPath, vector<TensorPath> tensorPaths,
          map<IndexExpr,TensorPath> mapReadNodesToPaths)
      : tensor(tensor),
        scheduleForest(scheduleForest),
        resultTensorPath(resultTensorPath),
        tensorPaths(tensorPaths),
        mapReadNodesToPaths(mapReadNodesToPaths) {}
  TensorBase                             tensor;
  IterationScheduleForest                scheduleForest;
  TensorPath                             resultTensorPath;
  vector<TensorPath>                     tensorPaths;
  map<IndexExpr,TensorPath>              mapReadNodesToPaths;
};

IterationSchedule::IterationSchedule() {
}

IterationSchedule IterationSchedule::make(const TensorBase& tensor) {
  IndexExpr expr = tensor.getExpr();

  vector<TensorPath> tensorPaths;

  // Create the tensor path formed by the result.
  vector<IndexVar> resultIndexVars;
  for (size_t i = 0; i < tensor.getOrder(); ++i) {
    size_t idx = tensor.getFormat().getModeOrder()[i];
    resultIndexVars.push_back(tensor.getIndexVars()[idx]);
  }
  TensorPath resultTensorPath = TensorPath(tensor, resultIndexVars);

  // Create the paths formed by tensor reads in the given expression.
  struct CollectTensorPaths : public expr_nodes::ExprVisitor {
    using ExprVisitor::visit;
    vector<TensorPath> tensorPaths;
    map<IndexExpr,TensorPath> mapReadNodesToPaths;
    void visit(const expr_nodes::ReadNode* op) {
      taco_iassert(op->tensor.getOrder() == op->indexVars.size()) <<
          "Tensor access " << IndexExpr(op) << " but tensor format only has " <<
          op->tensor.getOrder() << " dimensions.";
      Format format = op->tensor.getFormat();

      // copy index variables to path
      vector<IndexVar> path(op->indexVars.size());
      for (size_t i=0; i < op->indexVars.size(); ++i) {
        path[i] = op->indexVars[format.getModeOrder()[i]];
      }

      auto tensorPath = TensorPath(op->tensor, path);
      mapReadNodesToPaths.insert({op, tensorPath});
      tensorPaths.push_back(tensorPath);
    }
  };
  CollectTensorPaths collect;
  expr.accept(&collect);
  util::append(tensorPaths, collect.tensorPaths);
  map<IndexExpr,TensorPath> mapReadNodesToPaths = collect.mapReadNodesToPaths;

  // Construct a forest decomposition from the tensor path graph
  IterationScheduleForest forest =
      IterationScheduleForest(util::combine({resultTensorPath},tensorPaths));

  // Create the iteration schedule
  IterationSchedule schedule = IterationSchedule();
  schedule.content =
      make_shared<IterationSchedule::Content>(tensor,
                                              forest,
                                              resultTensorPath,
                                              tensorPaths,
                                              mapReadNodesToPaths);
  return schedule;
}

const TensorBase& IterationSchedule::getTensor() const {
  return content->tensor;
}

const std::vector<IndexVar>& IterationSchedule::getRoots() const {
  return content->scheduleForest.getRoots();
}

const IndexVar& IterationSchedule::getParent(const IndexVar& var) const {
  return content->scheduleForest.getParent(var);
}

const std::vector<IndexVar>&
IterationSchedule::getChildren(const IndexVar& var) const {
  return content->scheduleForest.getChildren(var);
}

vector<IndexVar> IterationSchedule::getAncestors(const IndexVar& var) const {
  std::vector<IndexVar> ancestors;
  ancestors.push_back(var);
  IndexVar parent = var;
  while (content->scheduleForest.hasParent(parent)) {
    parent = content->scheduleForest.getParent(parent);
    ancestors.push_back(parent);
  }
  return ancestors;
}

vector<IndexVar> IterationSchedule::getDescendants(const IndexVar& var) const{
  vector<IndexVar> descendants;
  descendants.push_back(var);
  for (auto& child : getChildren(var)) {
    util::append(descendants, getDescendants(child));
  }
  return descendants;
}

bool IterationSchedule::isLastFreeVariable(const IndexVar& var) const {
  return isFree(var) && !hasFreeVariableDescendant(var);
}

bool IterationSchedule::hasFreeVariableDescendant(const IndexVar& var) const {
  // Traverse the iteration schedule forest subtree of var to determine whether
  // it has any free variable descendants
  auto children = content->scheduleForest.getChildren(var);
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

bool
IterationSchedule::hasReductionVariableAncestor(const IndexVar& var) const {
  if (isReduction(var)) {
    return true;
  }

  IndexVar parent = var;
  while (content->scheduleForest.hasParent(parent)) {
    parent = content->scheduleForest.getParent(parent);
    if (isReduction(parent)) {
      return true;
    }
  }
  return false;
}

const vector<TensorPath>& IterationSchedule::getTensorPaths() const {
  return content->tensorPaths;
}

const TensorPath&
IterationSchedule::getTensorPath(const IndexExpr& operand) const {
  taco_iassert(util::contains(content->mapReadNodesToPaths, operand));
  return content->mapReadNodesToPaths.at(operand);
}

const TensorPath& IterationSchedule::getResultTensorPath() const {
  return content->resultTensorPath;
}

IndexVarType IterationSchedule::getIndexVarType(const IndexVar& var) const {
  return (util::contains(content->tensor.getIndexVars(), var))
      ? IndexVarType::Free : IndexVarType::Sum;
}

bool IterationSchedule::isFree(const IndexVar& var) const {
  return getIndexVarType(var) == IndexVarType::Free;
}

bool IterationSchedule::isReduction(const IndexVar& var) const {
  return !isFree(var);
}

std::ostream& operator<<(std::ostream& os, const IterationSchedule& schedule) {
  os << "Index Variable Forest" << std::endl;
  os << schedule.content->scheduleForest << std::endl;
  os << "Result tensor path" << std::endl;
  os << "  " << schedule.getResultTensorPath() << std::endl;
  os << "Tensor paths:" << std::endl;
  for (auto& tensorPath : schedule.getTensorPaths()) {
    os << "  " << tensorPath << std::endl;
  }
  return os;
}

}}
