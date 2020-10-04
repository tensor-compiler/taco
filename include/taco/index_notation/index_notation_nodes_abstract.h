#ifndef TACO_INDEX_NOTATION_NODES_ABSTRACT_H
#define TACO_INDEX_NOTATION_NODES_ABSTRACT_H

#include <vector>
#include <memory>

#include "taco/type.h"
#include "taco/util/uncopyable.h"
#include "taco/util/intrusive_ptr.h"

namespace taco {

class TensorVar;
class IndexVar;
class IndexExprVisitorStrict;
class IndexStmtVisitorStrict;
class OperatorSplit;
class Precompute;

/// A node of a scalar index expression tree.
struct IndexExprNode : public util::Manageable<IndexExprNode>,
                       private util::Uncopyable {
public:
  IndexExprNode();
  IndexExprNode(Datatype type);
  virtual ~IndexExprNode() = default;
  virtual void accept(IndexExprVisitorStrict*) const = 0;

  /// Return the scalar data type of the index expression.
  Datatype getDataType() const;

  /// Store the index expression's result to the given workspace w.r.t. index
  /// variable `i` and replace the index expression (in the enclosing
  /// expression) with a workspace access expression.  The index variable `i` is
  /// retained in the enclosing expression and used to access the workspace,
  /// while `iw` replaces `i` in the index expression that computes workspace
  /// results.
  void setWorkspace(IndexVar i, IndexVar iw, TensorVar workspace) const;

  /// Return a workspace scheduling construct that describes the workspace to
  /// store expression to.
  Precompute getWorkspace() const;

private:
  Datatype dataType;

  mutable std::shared_ptr<std::tuple<IndexVar,IndexVar,TensorVar>> workspace;
};


/// A node in a tensor index expression tree
struct IndexStmtNode : public util::Manageable<IndexStmtNode>,
                       private util::Uncopyable {
public:
  IndexStmtNode();
  IndexStmtNode(Type type);
  virtual ~IndexStmtNode() = default;
  virtual void accept(IndexStmtVisitorStrict*) const = 0;

  Type getType() const;

private:
  Type type;
};

}
#endif
