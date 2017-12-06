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
class Type;
class IndexExpr;
class ExprVisitorStrict;
struct AccessNode;

class TensorBase;

/// An index variable. Index variables are used in index expressions, where they
/// represent iteration over a tensor mode.
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

std::ostream& operator<<(std::ostream&, const IndexVar&);


/// A tensor variable in an index expression. Can be either an operand or the
/// result of the expression.
class TensorVar {
public:
  TensorVar();
  TensorVar(const Type& type);
  TensorVar(const std::string& name, const Type& type);

  std::string getName() const;
  const Type& getType() const;

private:
  struct Content;
  std::shared_ptr<Content> content;
};

std::ostream& operator<<(std::ostream&, const TensorVar&);


/// A node of an index expression tree.
struct ExprNode : public util::Manageable<ExprNode>, private util::Uncopyable {
public:
  virtual ~ExprNode() = default;
  virtual void accept(ExprVisitorStrict*) const = 0;
  virtual void print(std::ostream& os) const = 0;
};

/// An index expression.
class IndexExpr : public util::IntrusivePtr<const ExprNode> {
public:
  typedef ExprNode Node;

  IndexExpr() : util::IntrusivePtr<const ExprNode>(nullptr) {}
  IndexExpr(const ExprNode* n)
      : util::IntrusivePtr<const ExprNode>(n) {}

  IndexExpr(int);
  IndexExpr(float);
  IndexExpr(double);

  IndexExpr operator-();

  void accept(ExprVisitorStrict *) const;
  friend std::ostream& operator<<(std::ostream&, const IndexExpr&);
};

IndexExpr operator+(const IndexExpr&, const IndexExpr&);
IndexExpr operator-(const IndexExpr&, const IndexExpr&);
IndexExpr operator*(const IndexExpr&, const IndexExpr&);
IndexExpr operator/(const IndexExpr&, const IndexExpr&);


/// An index expression that represents a tensor access (e.g. A(i,j)).  Access
/// expressions are returned when calling the overloaded operator() on tensors
/// and can be assigned an expression.
class Access : public IndexExpr {
public:
  typedef AccessNode Node;

  Access() = default;
  Access(const Node* n);
  Access(const TensorBase& tensor, const std::vector<IndexVar>& indices={});

  const TensorBase &getTensor() const;
  const std::vector<IndexVar>& getIndexVars() const;

  /// Assign the result of an expression to a left-hand-side tensor access.
  void operator=(const IndexExpr&);
  void operator=(const Access&);

  /// Accumulate the result of an expression to a left-hand-side tensor access.
  void operator+=(const IndexExpr&);
  void operator+=(const Access&);

private:
  const Node* getPtr() const;
};

}
#endif
