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
  Expr(float);
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

struct IntImmNode : public internal::TENode {
  friend struct IntImm;

  IntImmNode(int val) : val(val) {}

  void print(std::ostream& os) const { os << val; }

  int val;
};

struct IntImm : public Expr {
  typedef IntImmNode Node;

  IntImm() = default;
  IntImm(const Node* n) : Expr(n) {}
  IntImm(int val) : IntImm(new Node(val)) {}

  const Node* getPtr() const {
    return static_cast<const Node*>(IntImm::ptr);
  }

  int getVal() const { return getPtr()->val; }
};

struct FloatImmNode : public internal::TENode {
  friend struct FloatImm;

  FloatImmNode(float val) : val(val) {}

  void print(std::ostream& os) const { os << val; }

  float val;
};

struct FloatImm : public Expr {
  typedef FloatImmNode Node;

  FloatImm() = default;
  FloatImm(const Node* n) : Expr(n) {}
  FloatImm(float val) : FloatImm(new Node(val)) {}

  const Node* getPtr() const {
    return static_cast<const Node*>(FloatImm::ptr);
  }

  float getVal() const { return getPtr()->val; }
};

struct DoubleImmNode : public internal::TENode {
  friend struct DoubleImm;

  DoubleImmNode(double val) : val(val) {}

  void print(std::ostream& os) const { os << val; }

  double val;
};

struct DoubleImm : public Expr {
  typedef DoubleImmNode Node;

  DoubleImm() = default;
  DoubleImm(const Node* n) : Expr(n) {}
  DoubleImm(double val) : DoubleImm(new Node(val)) {}

  const Node* getPtr() const {
    return static_cast<const Node*>(DoubleImm::ptr);
  }

  double getVal() const { return getPtr()->val; }
};

}
#endif
