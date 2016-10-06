#ifndef TACO_EXPR_H
#define TACO_EXPR_H

#include <iostream>
#include <string>

#include "error.h"
#include "util/intrusive_ptr.h"
#include "util/uncopyable.h"

namespace taco {
namespace internal {

template <typename Node>
struct TEHandle : public util::IntrusivePtr<const Node> {
  TEHandle() : util::IntrusivePtr<const Node>() {}
  TEHandle(const Node* n) : util::IntrusivePtr<const Node>(n) {}

  virtual ~TEHandle() = default;
};

struct TENode : private util::Uncopyable, public util::Manageable<TENode> {
  virtual void print(std::ostream& os) const { iassert(false); }

  template <typename Node>
  friend std::ostream& operator<<(std::ostream&, const TEHandle<Node>&);

  virtual ~TENode() = default;
};

}

template <typename Node>
std::ostream& operator<<(std::ostream& os, 
                         const internal::TEHandle<Node>& node) {
  node.ptr->print(os);
  return os;
}

class VarNode : public util::Manageable<VarNode> {
  friend struct Var;

  // The kind of an index variable.
  enum class Kind { Free, Reduction };

  VarNode(Kind, const std::string&);

  void print(std::ostream&) const;

  template <typename Node>
  friend std::ostream& operator<<(std::ostream&,
                                  const internal::TEHandle<Node>&);
  
  Kind        kind;
  std::string name;
};

struct Var : public internal::TEHandle<VarNode> {
  typedef VarNode Node;

  typedef Node::Kind Kind;

  static Kind Free;
  static Kind Reduction;

  Var(Kind = Kind::Free, const std::string& = "");
  Var(const std::string&, Kind = Kind::Free);
  
  const Node* getPtr() const {
    return static_cast<const Node*>(TEHandle<Node>::ptr);
  }
  
  Kind        getKind() const { return getPtr()->kind; }
  std::string getName() const { return getPtr()->name; }

private: 
  Var(const Node* c) : internal::TEHandle<VarNode>(c) {}
};

struct Expr : public internal::TEHandle<internal::TENode> {
  typedef internal::TENode Node;

  Expr() : internal::TEHandle<Node>() {}
  Expr(const Node* n) : internal::TEHandle<Node>(n) {}

  Expr(int);
  Expr(double);
  
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

  virtual void print(std::ostream& os) const { os << val; }

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

#if 0
struct Stmt : public internal::TEHandle<internal::TENode> {
  typedef internal::TENode Node;
  
  Stmt() : internal::TEHandle<Node>() {}
  Stmt(const Node* n) : internal::TEHandle<Node>(n) {}
  
  template <typename T> friend bool isa(Stmt);
  template <typename T> friend const T to(Stmt);
};

template <typename T>
inline bool isa(Stmt s) {
  return s.defined() && dynamic_cast<const typename T::Node*>(s.ptr) != nullptr;
}

template <typename T>
inline const T to(Stmt s) {
  iassert(isa<T>(s));
  return T(static_cast<const typename T::Node*>(s.ptr));
}
#endif

}

#endif
