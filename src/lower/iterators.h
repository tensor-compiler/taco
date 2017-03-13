#ifndef TACO_LOWER_ITERATORS_H
#define TACO_LOWER_ITERATORS_H

#include "iteration_schedule.h"
#include "tensor_path.h"
#include "storage/iterator.h"

#include <map>

namespace taco {
class TensorBase;

namespace ir {
class Expr;
}

namespace lower {

/// Tracks the per-edge iterators of the tensor paths of an iteration schedule.
class Iterators {
public:
  Iterators();

  Iterators(const IterationSchedule& schedule,
            const std::map<TensorBase,ir::Expr>& tensorVariables);

  /// Returns the root iterator.
  /// TODO: Should each path have a 0 step that's the root, so that we can use
  /// operator[] to get the root (with step 0)?
  const storage::Iterator& getRoot(const TensorPath&) const;

  /// Returns the iterator for the step.
  const storage::Iterator& operator[](const TensorPathStep&) const;

  /// Returns the iterators for the steps.
  std::vector<storage::Iterator>
  operator[](const std::vector<TensorPathStep>&) const;

private:
  std::map<TensorPath, storage::Iterator> roots;
  std::map<TensorPathStep, storage::Iterator> iterators;
};


/// Returns true iff the iterators must be merged, false otherwise. Iterators
/// must be merged iff two or more of them are not random access.
bool needsMerge(const std::vector<storage::Iterator>&);

/// Returns the dense iterators
std::vector<storage::Iterator>
getDenseIterators(const std::vector<storage::Iterator>&);

/// Returns the sequential access iterators
std::vector<storage::Iterator>
getSequentialAccessIterators(const std::vector<storage::Iterator>&);

/// Returns the random access iterators
std::vector<storage::Iterator>
getRandomAccessIterators(const std::vector<storage::Iterator>&);

/// Returns the idx vars of the iterators.
std::vector<ir::Expr> getIdxVars(const std::vector<storage::Iterator>&);

}}
#endif
