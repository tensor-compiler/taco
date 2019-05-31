#ifndef TACO_TENSOR_PATH_H
#define TACO_TENSOR_PATH_H

#include <memory>
#include <vector>

#include "taco/util/comparable.h"

namespace taco {

class TensorVar;
class IndexVar;
class Access;

class TensorPathStep;

/// A tensor Access expression such as A(i,j,k) results in a path in an
/// iteration graph through i,j,k. The exact path (i->j->k, j->k->i, etc.) is
/// dictated by the ordering of the levels in the tensor storage tree. The index
/// variable that indexes into the mode at the first level is the first index
/// variable in the path, and so forth.
class TensorPath : public util::Comparable<TensorPath> {
public:
  TensorPath();
  TensorPath(const std::vector<IndexVar>& path, const Access& access);

  /// Returns the Access expression that the path represents.
  const Access& getAccess() const;

  /// Returns the variables along the path.
  const std::vector<IndexVar>& getVariables() const;

  /// Returns the size (number of steps) of the path.
  size_t getSize() const;

  /// Returns the ith tensor step along the path.
  TensorPathStep getStep(size_t i) const;

  /// Returns the last step along this path.
  TensorPathStep getLastStep() const;

  /// Returns the step incident on var.
  TensorPathStep getStep(const IndexVar& var) const;

  /// True if the path is define, false otherwise
  bool defined() const;

  friend bool operator==(const TensorPath&, const TensorPath&);
  friend bool operator<(const TensorPath&, const TensorPath&);

private:
  struct Content;
  std::shared_ptr<Content> content;
};

std::ostream& operator<<(std::ostream&, const TensorPath&);


/// A step along a tensor path.
class TensorPathStep : public util::Comparable<TensorPathStep> {
public:
  /// Return the path the tensor belongs to.
  const TensorPath& getPath() const;

  /// Returns the location of this step in the path.
  int getStep() const;

private:
  TensorPath path;
  int step;

  TensorPathStep();
  TensorPathStep(const TensorPath& path, int step);
  friend TensorPath;
};

bool operator==(const TensorPathStep&, const TensorPathStep&);
bool operator<(const TensorPathStep&, const TensorPathStep&);
std::ostream& operator<<(std::ostream&, const TensorPathStep&);

}
#endif
