#ifndef TACO_ITERATION_GRAPH_H
#define TACO_ITERATION_GRAPH_H

#include <memory>
#include <vector>

namespace taco {

class TensorVar;
class IndexVar;
class IndexExpr;
class Assignment;

class TensorPath;

enum class IndexVarType {
  Free,
  Sum
};

/// An iteration graph constist of index variables arranged in a forest
/// together with tensor paths super-imposed on the forest.
/// - The iteration graph is arranged in a forest decomposition where all
///   tensor paths move from index variables higher in the tree to index
///   variables strictly lower in the tree.
/// - The tensor paths describe how to iterate over the index variables through
///   the indices of the corresponding (sparse or dense) tensors.
class IterationGraph {
public:
  IterationGraph();

  /// Creates an iteration graph for a tensor with a defined expression.
  static IterationGraph make(Assignment);

  /// Returns the iteration graph roots; the index variables with no parents.
  const std::vector<IndexVar>& getRoots() const;

  /// Returns the children of the index variable
  const std::vector<IndexVar>& getChildren(const IndexVar&) const;

  /// Returns the parent of the index variable
  const IndexVar& getParent(const IndexVar&) const;

  /// Returns the ancestors of the index variable including itself.
  std::vector<IndexVar> getAncestors(const IndexVar&) const;

  /// Returns all descendant of the index variable, including itself.
  std::vector<IndexVar> getDescendants(const IndexVar&) const;


  /// Returns the tensor paths of the operand tensors in the iteration graph.
  const std::vector<TensorPath>& getTensorPaths() const;

  /// Returns the tensor path corresponding to a tensor read expression.
  const TensorPath& getTensorPath(const IndexExpr&) const;

  /// Returns the tensor path of the result tensor.
  const TensorPath& getResultTensorPath() const;


  /// Returns the index variable type.
  IndexVarType getIndexVarType(const IndexVar&) const;

  /// Returns true iff the index variable is free.
  bool isFree(const IndexVar&) const;

  /// Returns true iff the index variable is a reduction.
  bool isReduction(const IndexVar&) const;

  /// Returns true if the index variable is the only free var in its subtree.
  bool isLastFreeVariable(const IndexVar&) const;

  /// Returns true if the index variable is the ancestor of any free variable.
  bool hasFreeVariableDescendant(const IndexVar&) const;

  /// Returns true if the index variable has a reduction variable ancestor.
  bool hasReductionVariableAncestor(const IndexVar&) const;


  /// Returns the index expression at the given index variable.
  const IndexExpr& getIndexExpr(const IndexVar&) const;


  /// Print an iteration graph as a dot file.
  void printAsDot(std::ostream&);

  /// Print an iteration graph.
  friend std::ostream& operator<<(std::ostream&, const IterationGraph&);

private:
  struct Content;
  std::shared_ptr<Content> content;
};

}
#endif
