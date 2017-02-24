#ifndef TACO_LOWER_ITERATORS_H
#define TACO_LOWER_ITERATORS_H

#include "iteration_schedule.h"
#include "tensor_path.h"
#include "var.h"
#include "storage/iterator.h"

#include <map>

namespace taco {

namespace internal {
class Tensor;
}

namespace ir {
class Expr;
}

namespace lower {

/// Tracks the per-edge iterators of the tensor paths of an iteration schedule.
class Iterators {
public:
  Iterators(const IterationSchedule& schedule,
            const std::map<internal::Tensor,ir::Expr>& tensorVariables);

  /// Returns the root iterator.
  const storage::Iterator& getRoot() const;

  /// Returns the iterator for the step.
  const storage::Iterator& operator[](const TensorPathStep&) const;

  /// Returns the iterators for the steps.
  std::vector<storage::Iterator>
  operator[](const std::vector<TensorPathStep>&) const;

private:
  storage::Iterator root;
  std::map<TensorPathStep, storage::Iterator> iterators;
};


/// Returns true iff the iterators must be merged, false otherwise. Iterators
/// must be merged iff two or more of them are not random access.
bool needsMerge(const std::vector<storage::Iterator>&);

/// Returns the iterators that are sequential access
std::vector<storage::Iterator>
getSequentialAccessIterators(const std::vector<storage::Iterator>&);

/// Returns the iterators that are random access
std::vector<storage::Iterator>
getRandomAccessIterators(const std::vector<storage::Iterator>&);

/// Returns the idx vars of the iterators.
std::vector<ir::Expr> getIdxVars(const std::vector<storage::Iterator>&);

}}
#endif
