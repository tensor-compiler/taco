#ifndef TACIT_OPERATOR_H
#define TACIT_OPERATOR_H

#include <iostream>
#include <string>

#include "expr.h"
#include "error.h"
#include "tensor.h"

namespace tacit {

struct NaryExpr;
struct Add;
struct Sub;
struct Mul;
struct Div;

template <typename CType> struct Read;

template <typename CType>
class ReadNode : public internal::TENode {
  friend struct Read<CType>;

  ReadNode(Tensor<CType> tensor, const std::vector<Var>& indices) : 
      tensor(tensor), indices(indices) {}

  virtual void print(std::ostream& os) const {
    os << tensor.getName() << "(" << util::join(indices) << ")";
  }

  Tensor<CType>    tensor;
  std::vector<Var> indices;
};

template <typename CType>
struct Read : public Expr {
  typedef ReadNode<CType> Node;

  Read() = default;
  Read(const Node* n) : Expr(n) {}
  Read(Tensor<CType> tensor, const std::vector<Var>& indices) : 
      Read(new Node(tensor, indices)) {}

  const Node* getPtr() const { return static_cast<const Node*>(Read::ptr); }

  Tensor<CType>    getTensor() const { return getPtr()->tensor; }
  std::vector<Var> getIndexVars() const { return getPtr()->indices; }

  void operator=(const Expr& source) { assign(source); }
  void operator=(const Read& source) { assign(source); }

private:
  void assign(Expr expr) {
    auto *tensor = const_cast<TensorObject<CType>*>(getPtr()->tensor.getPtr());
    uassert(!tensor->expr.defined()) << "Cannot reassign " << *tensor;

    tensor->indexVars = getIndexVars();
    tensor->expr = expr;
  }
};

class NaryExprNode : public internal::TENode {
  friend struct NaryExpr;

protected:
  NaryExprNode(const std::vector<Expr>& operands) : operands(operands) {}

  void printNary(std::ostream& os, const std::string& op) const {
    os << util::join(operands, op);
  }

  std::vector<Expr> operands;
  
private:
  // Syntactic sugar for arithmetic operations.
  friend Add operator+(const Expr&, const Expr&);
  friend Mul operator*(const Expr&, const Expr&);
  // friend Sub operator-(const Expr&, const Expr&);
  // friend Div operator/(const Expr&, const Expr&);
};

struct NaryExpr : public Expr {
  typedef NaryExprNode Node;

  NaryExpr() = default;
  NaryExpr(const Node* n) : Expr(n) {}
  NaryExpr(const std::vector<Expr>& operands) : 
      NaryExpr(new NaryExprNode(operands)) {}

  const Node* getPtr() const { return static_cast<const Node*>(Expr::ptr); }

  // Retrieve specified operand (casted to type E).
  template <typename E = Expr>
  E getOperand(size_t idx) const { return to<E>(getPtr()->operands[idx]); }
};

class AddNode : public NaryExprNode {
  friend struct Add;

  AddNode(const std::vector<Expr>& operands) : NaryExprNode(operands) {}
  
  virtual void print(std::ostream& os) const {
    printNary(os, " + ");
  }
};

struct Add : public NaryExpr {
  typedef AddNode Node;

  Add() = default;
  Add(const Node* n) : NaryExpr(n) {}
  Add(const std::vector<Expr>& operands) : Add(new AddNode(operands)) {}
};

class MulNode : public NaryExprNode {
  friend struct Mul;

  MulNode(const std::vector<Expr>& operands) : NaryExprNode(operands) {}
  
  virtual void print(std::ostream& os) const {
    printNary(os, " * ");
  }
};

struct Mul : public NaryExpr {
  typedef MulNode Node;

  Mul() = default;
  Mul(const Node* n) : NaryExpr(n) {}
  Mul(const std::vector<Expr>& operands) : Mul(new MulNode(operands)) {}
};

Add operator+(const Expr&, const Expr&);
Mul operator*(const Expr&, const Expr&);
// Sub operator-(const Expr&, const Expr&);
// Div operator/(const Expr&, const Expr&);

}

#endif
