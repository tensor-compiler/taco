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
class MergeRule;
class MergeLatticePoint;

/// An iteration schedule is a two dimensional ordering of index variables,
/// tensor paths that describe how to reach non-zero index variable values
/// through tensor indices, and a constraint on each index variable that tells
/// us how to merge tensor index values.
class IterationSchedule {
public:
  IterationSchedule();

  /// Creates an iteration schedule for a tensor with a defined expression.
  static IterationSchedule make(const internal::Tensor&);

  /// Returns the tensor the iteration schedule was built from.
  const internal::Tensor& getTensor() const;

  /// Returns the number of layers in the iteration schedule. Layers correspond
  /// to loop nests in the emitted code.s
  size_t numLayers() const;

  /// Returns a two dimensional ordering of index variables. The first (x)
  /// dimension corresponds to nested loops and the second (y) dimension
  /// correspond to sequenced loops.
  const std::vector<std::vector<taco::Var>>& getLayers() const;

  /// Returns the merge rule of the given var.
  const MergeRule& getMergeRule(const taco::Var&) const;

  /// Returns the tensor paths of the operand tensors in the iteration schedule.
  const std::vector<TensorPath>& getTensorPaths() const;

  /// Returns the tensor path corresponding to a tensor read expression operand.
  const TensorPath& getTensorPath(const taco::Expr& operand) const;

  /// Returns the tensor path of the result tensor.
  const TensorPath& getResultTensorPath() const;

  private:
  struct Content;
  std::shared_ptr<Content> content;
};

std::ostream& operator<<(std::ostream&, const IterationSchedule&);

}}
#endif
