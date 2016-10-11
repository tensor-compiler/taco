#ifndef TACO_OPERATOR_H
#define TACO_OPERATOR_H

#include <iostream>
#include <string>

#include "expr.h"
#include "expr_nodes.h"
#include "error.h"
#include "util/strings.h"

// TODO: delete
#include "expr_visitor.h"

namespace taco {
namespace internal {
class Tensor;

template <typename T>
std::vector<Expr> mergeOperands(const Expr&, const Expr&);
}  // namespace internal

class Var;

struct Read : public Expr {
  typedef internal::ReadNode Node;

  Read() = default;
  Read(const Node* n);
  Read(const internal::Tensor& tensor, const std::vector<Var>& indices);

  const Node* getPtr() const;

  const internal::Tensor &getTensor() const;

  const std::vector<Var>& getIndexVars() const;

  void operator=(const Expr& source) {
    assign(source);
  }
  
  void operator=(const Read& source) {
    assign(source);
  }

private:
  void assign(Expr);
};

struct NaryExpr : public Expr {
  typedef internal::NaryExprNode Node;

  NaryExpr() = default;
  NaryExpr(const Node* n) : Expr(n) {}

  const Node* getPtr() const { return static_cast<const Node*>(Expr::ptr); }

  // Retrieve specified operand (casted to type E).
  template <typename E = Expr>
  E getOperand(size_t idx) const { return to<E>(getPtr()->operands[idx]); }
};

struct BinaryExpr : public Expr {
  BinaryExpr() = default;
  BinaryExpr(const Node* n) : Expr(n) {}

  const internal::BinaryExprNode* getPtr() const {
    return static_cast<const internal::BinaryExprNode*>(Expr::ptr);
  }

  // Retrieve left operand (casted to type E).
  template <typename E = Expr>
  E getLhs() const {
    return to<E>(getPtr()->lhs);
  }

  // Retrieve right operand (casted to type E).
  template <typename E = Expr>
  E getRhs() const {
    return to<E>(getPtr()->rhs);
  }
};

struct Add : public NaryExpr {
  typedef internal::AddNode Node;

  Add() = default;
  Add(const internal::AddNode* n) : NaryExpr(n) {}
  Add(const std::vector<Expr>& operands) : Add(new Node(operands)) {}
};

struct Sub : public BinaryExpr {
  typedef internal::SubNode Node;

  Sub() = default;
  Sub(const Node* n) : BinaryExpr(n) {}
  Sub(Expr lhs, Expr rhs) : Sub(new Node(lhs, rhs)) {}
};

struct Mul : public NaryExpr {
  typedef internal::MulNode Node;

  Mul() = default;
  Mul(const internal::MulNode* n) : NaryExpr(n) {}
  Mul(const std::vector<Expr>& operands) : Mul(new Node(operands)) {}
};

struct Div : public BinaryExpr {
  typedef internal::DivNode Node;

  Div() = default;
  Div(const Node* n) : BinaryExpr(n) {}
  Div(Expr lhs, Expr rhs) : Div(new Node(lhs, rhs)) {}
};

struct IntImm : public Expr {
  typedef internal::IntImmNode Node;

  IntImm() = default;
  IntImm(const Node* n) : Expr(n) {}
  IntImm(int val) : IntImm(new Node(val)) {}

  const Node* getPtr() const {
    return static_cast<const Node*>(IntImm::ptr);
  }

  int getVal() const { return getPtr()->val; }
};

struct FloatImm : public Expr {
  typedef internal::FloatImmNode Node;

  FloatImm() = default;
  FloatImm(const Node* n) : Expr(n) {}
  FloatImm(float val) : FloatImm(new Node(val)) {}

  const Node* getPtr() const {
    return static_cast<const Node*>(FloatImm::ptr);
  }

  float getVal() const { return getPtr()->val; }
};

struct DoubleImm : public Expr {
  typedef internal::DoubleImmNode Node;

  DoubleImm() = default;
  DoubleImm(const Node* n) : Expr(n) {}
  DoubleImm(double val) : DoubleImm(new Node(val)) {}

  const Node* getPtr() const {
    return static_cast<const Node*>(DoubleImm::ptr);
  }

  double getVal() const { return getPtr()->val; }
};

Add operator+(const Expr&, const Expr&);
Sub operator-(const Expr&, const Expr&);
Mul operator*(const Expr&, const Expr&);
Div operator/(const Expr&, const Expr&);

}
#endif
