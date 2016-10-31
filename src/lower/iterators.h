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

  /// Returns the iterator for the step.
  const storage::Iterator& getIterator(const TensorPathStep& step) const;

  /// Returns the iteration of the previous step in the path. If there are no
  /// previous steps then the identity iterator is returned.
  const storage::Iterator& getParentIterator(const TensorPathStep& step) const;

  const storage::Iterator& getResultIterator(const taco::Var& var) const;

private:
  std::map<TensorPathStep, storage::Iterator> iterators;
  std::map<taco::Var, storage::Iterator> resultIterators;
};

}}
#endif
