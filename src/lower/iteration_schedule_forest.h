#ifndef TACO_ITERATION_SCHEDULE_FOREST_H
#define TACO_ITERATION_SCHEDULE_FOREST_H

#include <vector>
#include <map>

#include "var.h"
#include "tensor_path.h"

namespace taco {
namespace lower {

/// An iteration schedule forest is a tree decomposition of a tensor path graph,
/// where all tensor path edges move from an index variable higher in the tree
/// to one strictly lower in the tree.
class IterationScheduleForest {
public:
  IterationScheduleForest() {}

  IterationScheduleForest(const std::vector<TensorPath>& paths);

  const std::vector<Var>& getRoots() const {return roots;}

  bool hasParent(const Var&) const;

  const Var& getParent(const Var&) const;

  const std::vector<Var>& getChildren(const Var&) const;

  std::vector<Var> getNodes() const;

  friend std::ostream& operator<<(std::ostream&,const IterationScheduleForest&);

private:
  std::vector<Var>                roots;
  std::map<Var, std::vector<Var>> children;
  std::map<Var, Var>              parents;
};

}}
#endif
