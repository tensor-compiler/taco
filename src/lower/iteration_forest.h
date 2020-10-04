#ifndef TACO_ITERATION_FOREST_H
#define TACO_ITERATION_FOREST_H

#include <vector>
#include <map>

#include "taco/index_notation/index_notation.h"
#include "tensor_path.h"

namespace taco {

/// An iteration forest is a tree decomposition of a tensor path graph, where
/// all tensor path edges move from an index variable higher in the tree to one
/// strictly lower in the tree.
class IterationForest {
public:
  IterationForest() {}

  IterationForest(const std::vector<TensorPath>& paths);

  const std::vector<IndexVar>& getRoots() const {return roots;}

  bool hasParent(const IndexVar&) const;

  const IndexVar& getParent(const IndexVar&) const;

  const std::vector<IndexVar>& getChildren(const IndexVar&) const;

  std::vector<IndexVar> getNodes() const;

  friend std::ostream& operator<<(std::ostream&,const IterationForest&);

private:
  std::vector<IndexVar>                     roots;
  std::map<IndexVar, std::vector<IndexVar>> children;
  std::map<IndexVar, IndexVar>              parents;
};

}
#endif
