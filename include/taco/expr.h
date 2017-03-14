#ifndef TACO_EXPR_H
#define TACO_EXPR_H

#include <iostream>
#include <string>
#include <typeinfo>
#include <memory>

#include "taco/util/error.h"
#include "taco/util/intrusive_ptr.h"
#include "taco/util/comparable.h"
#include "taco/util/uncopyable.h"

namespace taco {

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


namespace internal {
class ExprVisitorStrict;

struct ExprNode : public util::Manageable<ExprNode>, private util::Uncopyable {
  virtual ~ExprNode() = default;
  virtual void accept(ExprVisitorStrict*) const = 0;
  virtual void print(std::ostream& os) const = 0;
};
}

/// An index expression.
class Expr : public util::IntrusivePtr<const internal::ExprNode> {
public:

  Expr() : util::IntrusivePtr<const internal::ExprNode>(nullptr) {}
  Expr(const internal::ExprNode* n)
      : util::IntrusivePtr<const internal::ExprNode>(n) {}

  Expr(int);
  Expr(float);
  Expr(double);

  Expr operator-();

  void accept(internal::ExprVisitorStrict *) const;
  friend std::ostream& operator<<(std::ostream&, const Expr&);
};

/// Returns true if expression e is of type E
template <typename E>
inline bool isa(Expr e) {
  return e.defined() && dynamic_cast<const typename E::Node*>(e.ptr) != nullptr;
}

/// Casts the expression e to type E
template <typename E>
inline const E to(Expr e) {
  iassert(isa<E>(e)) << "Cannot convert " << typeid(e).name()
                     << " to " << typeid(E).name();
  return E(static_cast<const typename E::Node*>(e.ptr));
}

}
#endif
