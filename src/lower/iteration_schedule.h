#ifndef TACO_ITERATION_SCHEDULE_H
#define TACO_ITERATION_SCHEDULE_H

#include <memory>
#include <vector>

namespace taco {
class Var;
class Expr;

namespace internal {
class Tensor;
}

namespace lower {
class TensorPath;

/// An iteration schedule is a set of index variables arranged as a forest,
/// a set of tensor paths super-imposed on the forest.
/// - The iteration schedule is arranged in a forest decomposition where all
///   tensor paths move from index variables higher in the tree to index
///   variables strictly lower in the tree.
/// - The tensor paths describe how to iterate over the index variables through
///   the indices of the corresponding (sparse or dense) tensors.
class IterationSchedule {
public:
  IterationSchedule();

  /// Creates an iteration schedule for a tensor with a defined expression.
  static IterationSchedule make(const internal::Tensor&);

  /// Returns the tensor the iteration schedule was built from.
  const internal::Tensor& getTensor() const;

  /// Returns the iteration schedule roots; the index variables with no parents.
  const std::vector<taco::Var>&  getRoots() const;

  /// Returns the children of the index variable
  const std::vector<taco::Var>& getChildren(const taco::Var&) const;

  /// Returns true if the index variable is the ancestor of any free variable.
  bool hasFreeVariableDescendant(const taco::Var&) const;

  /// Returns true if the index variable is the only free var in its subtree
  bool isLastFreeVariable(const taco::Var&) const;

  /// Returns true if the index variable has a reduction variable ancestor.
  bool hasReductionVariableAncestor(const taco::Var&) const;

  /// Returns the tensor paths of the operand tensors in the iteration schedule.
  const std::vector<TensorPath>& getTensorPaths() const;

  /// Returns the tensor path corresponding to a tensor read expression.
  const TensorPath& getTensorPath(const taco::Expr&) const;

  /// Returns the tensor path of the result tensor.
  const TensorPath& getResultTensorPath() const;

  friend std::ostream& operator<<(std::ostream&, const IterationSchedule&);

private:
  struct Content;
  std::shared_ptr<Content> content;
};

}}
#endif
