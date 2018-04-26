#ifndef TACO_EXPR_NODE_H
#define TACO_EXPR_NODE_H

#include <vector>
#include <memory>

#include "taco/type.h"
#include "taco/util/uncopyable.h"
#include "taco/util/intrusive_ptr.h"

namespace taco {

class IndexVar;
class ExprVisitorStrict;
class OperatorSplit;

/// A node of an index expression tree.
struct ExprNode : public util::Manageable<ExprNode>, private util::Uncopyable {
public:
  ExprNode();
  ExprNode(DataType type);
  virtual ~ExprNode() = default;
  virtual void accept(ExprVisitorStrict*) const = 0;

  /// Split the expression.
  void splitOperator(IndexVar old, IndexVar left, IndexVar right);

  /// Returns the expression's operator splits.
  const std::vector<OperatorSplit>& getOperatorSplits() const;

  DataType getDataType() const;

private:
  std::shared_ptr<std::vector<OperatorSplit>> operatorSplits;
  DataType dataType;
};


///
struct TensorExprNode : public util::Manageable<TensorExprNode>,
                        private util::Uncopyable {
public:
  TensorExprNode();
  TensorExprNode(Type type);
  virtual ~TensorExprNode() = default;

private:

};

}
#endif
