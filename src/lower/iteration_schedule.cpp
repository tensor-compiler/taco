#include "iteration_schedule.h"

#include <set>
#include <vector>
#include <queue>

#include "iteration_schedule_forest.h"
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
  Content(internal::Tensor tensor,
          IterationScheduleForest scheduleForest,
          TensorPath resultTensorPath,
          vector<TensorPath> tensorPaths,
          map<Var,MergeRule> mergeRules,
          map<Expr,TensorPath> mapReadNodesToPaths)
      : tensor(tensor),
        scheduleForest(scheduleForest),
        resultTensorPath(resultTensorPath),
        tensorPaths(tensorPaths),
        mergeRules(mergeRules),
        mapReadNodesToPaths(mapReadNodesToPaths) {}

  internal::Tensor        tensor;

  IterationScheduleForest scheduleForest;

  TensorPath              resultTensorPath;
  vector<TensorPath>      tensorPaths;

  map<Var,MergeRule>      mergeRules;
  map<Expr,TensorPath>    mapReadNodesToPaths;
};

IterationSchedule::IterationSchedule() {
}

static map<Var,MergeRule> createMergeRules(const internal::Tensor& tensor,
                                           vector<Var> indexVariables,
                                           map<Expr,TensorPath> tensorPaths,
                                           const TensorPath& resultTensorPath) {
  map<Var,MergeRule> mergeRules;
  for (auto& indexVar : indexVariables) {
    mergeRules.insert({indexVar, MergeRule::make(tensor.getExpr(), indexVar,
                                            tensorPaths)});
  }
  return mergeRules;
}

IterationSchedule IterationSchedule::make(const internal::Tensor& tensor) {
  Expr expr = tensor.getExpr();

  vector<TensorPath> tensorPaths;

  // Create the tensor path formed by the result.
  TensorPath resultTensorPath = (tensor.getOrder())
                                ? TensorPath(tensor, tensor.getIndexVars())
                                : TensorPath();
  if (resultTensorPath.defined()) {
    tensorPaths.push_back(resultTensorPath);
  }

  // Create the paths formed by tensor reads in the given expression.
  struct CollectTensorPaths : public internal::ExprVisitor {
    using ExprVisitor::visit;
    vector<TensorPath> tensorPaths;
    map<Expr,TensorPath> mapReadNodesToPaths;
    void visit(const internal::Read* op) {
      // Scalars don't have a path
      if (op->tensor.getOrder() == 0) return;

      Format format = op->tensor.getFormat();
      vector<Var> path(op->indexVars.size());
      for (size_t i=0; i < op->indexVars.size(); ++i) {
        path[format.getLevels()[i].getDimension()] = op->indexVars[i];
      }

      auto tensorPath = TensorPath(op->tensor, path);
      mapReadNodesToPaths.insert({op, tensorPath});
      tensorPaths.push_back(tensorPath);
    }
  };
  CollectTensorPaths collect;
  expr.accept(&collect);
  util::append(tensorPaths, collect.tensorPaths);
  map<Expr,TensorPath> mapReadNodesToPaths = collect.mapReadNodesToPaths;

  // Construct a forest decomposition from the tensor path graph
  IterationScheduleForest forest = IterationScheduleForest(tensorPaths);

  // Create merge rules that describe how to merge the tensor paths incomming
  // on each index variable.
  map<Var,MergeRule> mergeRules = createMergeRules(tensor, forest.getNodes(),
                                                   mapReadNodesToPaths,
                                                   resultTensorPath);

  // Create the iteration schedule
  IterationSchedule schedule = IterationSchedule();
  schedule.content =
      make_shared<IterationSchedule::Content>(tensor,
                                              forest,
                                              resultTensorPath,
                                              tensorPaths,
                                              mergeRules,
                                              mapReadNodesToPaths);
  return schedule;
}

const internal::Tensor& IterationSchedule::getTensor() const {
  return content->tensor;
}

const std::vector<taco::Var>& IterationSchedule::getRoots() const {
  return content->scheduleForest.getRoots();
}

const std::vector<taco::Var>&
IterationSchedule::getChildren(const taco::Var& var) const {
  return content->scheduleForest.getChildren(var);
}

bool IterationSchedule::hasFreeVariableDescendant(const taco::Var& var) const {
  // Traverse the iteration schedule forest subtree of var to determine whether
  // it has any free variable descendants
  auto children = content->scheduleForest.getChildren(var);
  for (auto& child : children) {
    if (child.isFree()) {
      return true;
    }
    // Child is not free; check if it a free descendent
    if (hasFreeVariableDescendant(child)) {
      return true;
    }
  }
  return false;
}

bool IterationSchedule::isLastFreeVariable(const taco::Var& var) const {
  return var.isFree() && !hasFreeVariableDescendant(var);
}

bool
IterationSchedule::hasReductionVariableAncestor(const taco::Var& var) const {
  Var parent = var;
  while (content->scheduleForest.hasParent(parent)) {
    parent = content->scheduleForest.getParent(parent);
    if (parent.isReduction()) {
      return true;
    }
  }
  return false;
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
  os << "Index Variable Forest" << std::endl;
  os << schedule.content->scheduleForest << std::endl;
  os << "Merge rules:" << std::endl;
  for (auto& var : schedule.content->scheduleForest.getNodes()) {
    os << "  " << var << ": " << schedule.getMergeRule(var) << std::endl;
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
