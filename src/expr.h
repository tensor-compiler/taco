#ifndef TACO_EXPR_H
#define TACO_EXPR_H

#include <iostream>
#include <string>

#include "error.h"
#include "util/intrusive_ptr.h"
#include "util/uncopyable.h"

namespace taco {
namespace internal {

class ExprVisitor;

struct TENode : private util::Uncopyable, public util::Manageable<TENode> {
  virtual ~TENode() = default;
//  virtual void accept(ExprVisitor*) const = 0;
  virtual void accept(ExprVisitor*) const {}
  virtual void print(std::ostream& os) const = 0;
};
}  // namespace internal


class Expr : public util::IntrusivePtr<const internal::TENode> {
public:
  typedef internal::TENode Node;

  Expr() : util::IntrusivePtr<const internal::TENode>() {}
  Expr(const Node* n) : util::IntrusivePtr<const internal::TENode>(n) {}

  Expr(int);
  Expr(double);

  void accept(internal::ExprVisitor *) const;

  friend std::ostream& operator<<(std::ostream& os, const Expr& expr) {
    expr.ptr->print(os);
    return os;
  }

  template <typename T> friend bool isa(Expr);
  template <typename T> friend const T to(Expr);
};

template <typename T>
inline bool isa(Expr e) {
  return e.defined() && dynamic_cast<const typename T::Node*>(e.ptr) != nullptr;
}

template <typename T>
inline const T to(Expr e) {
  iassert(isa<T>(e)) << "Cannot convert " << typeid(e).name()
                     << " to " << typeid(T).name();
  return T(static_cast<const typename T::Node*>(e.ptr));
}

template <typename CType> struct Imm;

template <typename CType>
struct ImmNode : public internal::TENode {
  friend struct Imm<CType>;

  ImmNode(CType val) : val(val) {}

  void print(std::ostream& os) const { os << val; }

  CType val;
};

template <typename CType>
struct Imm : public Expr {
  typedef ImmNode<CType> Node;

  Imm() = default;
  Imm(const Node* n) : Expr(n) {}
  Imm(CType val) : Imm(new Node(val)) {}

  const Node* getPtr() const { return static_cast<const Node*>(Imm::ptr); }

  CType getVal() const { return getPtr()->val; }
};

}
#endif
