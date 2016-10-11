#ifndef TACO_ITERATION_SCHEDULE_H
#define TACO_ITERATION_SCHEDULE_H

#include <memory>
#include <vector>
#include <map>
#include <set>

namespace taco {
class Var;
class Expr;

namespace internal {
class Tensor;

/// A tensor Read expression such as A(i,j,k) results in a path in an iteration
/// schedule through i,j,k. The exact path (i->j->k, j->k->i, etc.) is dictated
/// by the order of the levels in the tensor storage tree. The index variable
/// that indexes into the dimension at the first level is the first index
/// variable in the path, and so forth.
class TensorPath {
public:
  TensorPath(Tensor tensor, std::vector<Var> path);
  const Tensor& getTensor() const;
  const std::vector<Var>& getPath() const;

private:
  struct Content;
  std::shared_ptr<Content> content;
};

std::ostream& operator<<(std::ostream&, const TensorPath&);


/// An iteration schedule is a two dimensional ordering of index variables,
/// tensor paths that describe how to reach non-zero index variable values
/// through tensor indices, and a constraint on each index variable that tells
/// us how to merge tensor index values.
class IterationSchedule {
public:
  IterationSchedule();
  static IterationSchedule make(const taco::Expr& expr);

  /// Return a two dimensional ordering of index variables. The first (x)
  /// dimension corresponds to nested loops and the second (y) dimension
  /// correspond to sequenced loops.
  const std::vector<std::vector<taco::Var>>& getIndexVariables() const;

  /// Return the tensor paths of the iteration schedule
  const std::vector<TensorPath>& getTensorPaths() const;

private:
  struct Content;
  std::shared_ptr<Content> content;
};

std::ostream& operator<<(std::ostream&, const IterationSchedule&);

}}
#endif
