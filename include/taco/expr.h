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
class Var : public util::Comparable<Var> {
public:
  enum Kind { Free, Sum };

private:
  struct Content {
    Var::Kind   kind;
    std::string name;
  };

public:
  Var(Kind kind = Kind::Free);
  Var(const std::string& name, Kind kind = Kind::Free);

  std::string getName() const {return content->name;}

  Kind getKind() const {return content->kind;}

  bool isFree() const {return content->kind == Free;}

  bool isReduction() const {return content->kind != Free;}

  friend bool operator==(const Var& l, const Var& r) {
    return l.content == r.content;
  }

  friend bool operator<(const Var& l, const Var& r) {
    return l.content < r.content;
  }

private:
  std::shared_ptr<Content> content;
};

std::ostream& operator<<(std::ostream& os, const Var& var);


namespace expr_nodes {
/// A node of an index expression tree.
struct ExprNode : public util::Manageable<ExprNode>, private util::Uncopyable {
  virtual ~ExprNode() = default;
  virtual void accept(ExprVisitorStrict*) const = 0;
  virtual void print(std::ostream& os) const = 0;
};
}

/// An index expression.
class Expr : public util::IntrusivePtr<const expr_nodes::ExprNode> {
public:
  typedef expr_nodes::ExprNode Node;

  Expr() : util::IntrusivePtr<const expr_nodes::ExprNode>(nullptr) {}
  Expr(const expr_nodes::ExprNode* n)
      : util::IntrusivePtr<const expr_nodes::ExprNode>(n) {}

  Expr(int);
  Expr(float);
  Expr(double);

  Expr operator-();

  void accept(expr_nodes::ExprVisitorStrict *) const;
  friend std::ostream& operator<<(std::ostream&, const Expr&);
};


/// An index expression that represents a tensor access (e.g. A(i,j)).  Access
/// expressions are returned when calling the overloaded operator() on tensors
/// and can be assigned an expression.
class Access : public Expr {
public:
  typedef expr_nodes::ReadNode Node;

  Access() = default;
  Access(const Node* n);
  Access(const TensorBase& tensor, const std::vector<Var>& indices={});

  const TensorBase &getTensor() const;
  const std::vector<Var>& getIndexVars() const;

  /// Assign an expression to a left-hand-side tensor access.
  void operator=(const Expr&  expr);

private:
  const Node* getPtr() const;
  void assign(Expr);
};

Expr operator+(const Expr&, const Expr&);
Expr operator-(const Expr&, const Expr&);
Expr operator*(const Expr&, const Expr&);
Expr operator/(const Expr&, const Expr&);

std::vector<TensorBase> getOperands(Expr expr);

}
#endif
