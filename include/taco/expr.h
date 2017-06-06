#ifndef TACO_EXPR_H
#define TACO_EXPR_H

#include <iostream>
#include <string>
#include <typeinfo>
#include <memory>
#include <vector>

#include "taco/error.h"
#include "taco/util/intrusive_ptr.h"
#include "taco/util/comparable.h"
#include "taco/util/uncopyable.h"

namespace taco {
class TensorBase;
namespace expr_nodes {
struct ReadNode;
class ExprVisitorStrict;
}

/// An index variable. Index variables are used in index expressions, where they
/// represent iteration over a tensor dimension.
class IndexVar : public util::Comparable<IndexVar> {
public:
  IndexVar();
  IndexVar(const std::string& name);

  std::string getName() const;
  friend bool operator==(const IndexVar&, const IndexVar&);
  friend bool operator<(const IndexVar&, const IndexVar&);

private:
  struct Content;
  std::shared_ptr<Content> content;
};

std::ostream& operator<<(std::ostream& os, const IndexVar& var);


namespace expr_nodes {
/// A node of an index expression tree.
struct ExprNode : public util::Manageable<ExprNode>, private util::Uncopyable {
  virtual ~ExprNode() = default;
  virtual void accept(ExprVisitorStrict*) const = 0;
  virtual void print(std::ostream& os) const = 0;
};
}

/// An index expression.
class IndexExpr : public util::IntrusivePtr<const expr_nodes::ExprNode> {
public:
  typedef expr_nodes::ExprNode Node;

  IndexExpr() : util::IntrusivePtr<const expr_nodes::ExprNode>(nullptr) {}
  IndexExpr(const expr_nodes::ExprNode* n)
      : util::IntrusivePtr<const expr_nodes::ExprNode>(n) {}

  IndexExpr(int);
  IndexExpr(float);
  IndexExpr(double);

  IndexExpr operator-();

  void accept(expr_nodes::ExprVisitorStrict *) const;
  friend std::ostream& operator<<(std::ostream&, const IndexExpr&);
};


/// An index expression that represents a tensor access (e.g. A(i,j)).  Access
/// expressions are returned when calling the overloaded operator() on tensors
/// and can be assigned an expression.
class Access : public IndexExpr {
public:
  typedef expr_nodes::ReadNode Node;

  Access() = default;
  Access(const Node* n);
  Access(const TensorBase& tensor, const std::vector<IndexVar>& indices={});

  const TensorBase &getTensor() const;
  const std::vector<IndexVar>& getIndexVars() const;

  /// Assign an expression to a left-hand-side tensor access.
  void operator=(const IndexExpr&);
  void operator=(const Access&);

private:
  const Node* getPtr() const;
  void assign(const IndexExpr&);
};

IndexExpr operator+(const IndexExpr&, const IndexExpr&);
IndexExpr operator-(const IndexExpr&, const IndexExpr&);
IndexExpr operator*(const IndexExpr&, const IndexExpr&);
IndexExpr operator/(const IndexExpr&, const IndexExpr&);

}
#endif
