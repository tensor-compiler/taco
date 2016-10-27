#ifndef TACO_TENSOR_PATH_H
#define TACO_TENSOR_PATH_H

#include <memory>
#include <vector>

#include "util/comparable.h"

namespace taco {
class Var;

namespace internal {
class Tensor;
}

namespace is {

/// A tensor Read expression such as A(i,j,k) results in a path in an iteration
/// schedule through i,j,k. The exact path (i->j->k, j->k->i, etc.) is dictated
/// by the order of the levels in the tensor storage tree. The index variable
/// that indexes into the dimension at the first level is the first index
/// variable in the path, and so forth.
class TensorPath : public util::Comparable<TensorPath> {
public:
  TensorPath();
  TensorPath(internal::Tensor tensor, std::vector<Var> path);

  /// Returns the tensor whose read created a path in the iteration schedule.
  const internal::Tensor& getTensor() const;

  /// Returns an iteration schedule path created by a tensor read.
  const std::vector<Var>& getVariables() const;

  friend bool operator==(const TensorPath&, const TensorPath&);
  friend bool operator<(const TensorPath&, const TensorPath&);

private:
  struct Content;
  std::shared_ptr<Content> content;
};

std::ostream& operator<<(std::ostream&, const TensorPath&);


/// A step (location) in a tensor path.
class TensorPathStep : public util::Comparable<TensorPathStep> {
public:
  TensorPathStep();
  TensorPathStep(const TensorPath& path, size_t step);

  const TensorPath& getPath() const;
  size_t getStep() const;

  friend bool operator==(const TensorPathStep&, const TensorPathStep&);
  friend bool operator<(const TensorPathStep&, const TensorPathStep&);

private:
  struct Content;
  std::shared_ptr<Content> content;
};

std::ostream& operator<<(std::ostream&, const TensorPathStep&);

}}
#endif
