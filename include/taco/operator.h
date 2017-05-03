#ifndef TACO_OPERATOR_H
#define TACO_OPERATOR_H

#include <iostream>
#include <string>

#include "taco/expr.h"
#include "taco/expr_nodes/expr_nodes.h"
#include "taco/util/error.h"
#include "taco/util/strings.h"

namespace taco {
class TensorBase;
class Var;

class Read : public Expr {
public:
  typedef expr_nodes::ReadNode Node;

  Read() = default;
  Read(const Node* n);
  Read(const TensorBase& tensor);
  Read(const TensorBase& tensor, const std::vector<Var>& indices);

  const TensorBase &getTensor() const;

  const std::vector<Var>& getIndexVars() const;

  void operator=(const Expr&  expr);

private:
  const Node* getPtr() const;

  void assign(Expr);
};

class UnaryExpr : public Expr {
public:
  typedef expr_nodes::UnaryExprNode Node;
  
  UnaryExpr() = default;
  UnaryExpr(const Node* n) : Expr(n) {}

  // Retrieve operand (casted to type E).
  template <typename E = Expr>
  E getOperand() const {
    return to<E>(getPtr()->a);
  }

private:
  const Node* getPtr() const {
    return static_cast<const Node*>(ptr);
  }
};

class BinaryExpr : public Expr {
public:
  typedef expr_nodes::BinaryExprNode Node;

  BinaryExpr() = default;
  BinaryExpr(const Node* n) : Expr(n) {}

  // Retrieve left operand (casted to type E).
  template <typename E = Expr>
  E getLhs() const {
    return to<E>(getPtr()->a);
  }

  // Retrieve right operand (casted to type E).
  template <typename E = Expr>
  E getRhs() const {
    return to<E>(getPtr()->b);
  }

private:
  const Node* getPtr() const {
    return static_cast<const Node*>(ptr);
  }
};

class Neg : public UnaryExpr {
public:
  typedef expr_nodes::NegNode Node;

  Neg() = default;
  Neg(const Node* n) : UnaryExpr(n) {}
  Neg(Expr operand) : Neg(new Node(operand)) {}
};

class Sqrt : public UnaryExpr {
public:
  typedef expr_nodes::SqrtNode Node;

  Sqrt() = default;
  Sqrt(const Node* n) : UnaryExpr(n) {}
  Sqrt(Expr operand) : Sqrt(new Node(operand)) {}
};

class Add : public BinaryExpr {
public:
  typedef expr_nodes::AddNode Node;

  Add() = default;
  Add(const Node* n) : BinaryExpr(n) {}
  Add(Expr lhs, Expr rhs) : Add(new Node(lhs, rhs)) {}
};

class Sub : public BinaryExpr {
public:
  typedef expr_nodes::SubNode Node;

  Sub() = default;
  Sub(const Node* n) : BinaryExpr(n) {}
  Sub(Expr lhs, Expr rhs) : Sub(new Node(lhs, rhs)) {}
};

class Mul : public BinaryExpr {
public:
  typedef expr_nodes::MulNode Node;

  Mul() = default;
  Mul(const Node* n) : BinaryExpr(n) {}
  Mul(Expr lhs, Expr rhs) : Mul(new Node(lhs, rhs)) {}
};

class Div : public BinaryExpr {
public:
  typedef expr_nodes::DivNode Node;

  Div() = default;
  Div(const Node* n) : BinaryExpr(n) {}
  Div(Expr lhs, Expr rhs) : Div(new Node(lhs, rhs)) {}
};

class IntImm : public Expr {
public:
  typedef expr_nodes::IntImmNode Node;

  IntImm() = default;
  IntImm(const Node* n) : Expr(n) {}
  IntImm(int val) : IntImm(new Node(val)) {}

  int getVal() const { return getPtr()->val; }

private:
  const Node* getPtr() const {
    return static_cast<const Node*>(ptr);
  }
};

class FloatImm : public Expr {
public:
  typedef expr_nodes::FloatImmNode Node;

  FloatImm() = default;
  FloatImm(const Node* n) : Expr(n) {}
  FloatImm(float val) : FloatImm(new Node(val)) {}

  float getVal() const { return getPtr()->val; }

private:
  const Node* getPtr() const {
    return static_cast<const Node*>(ptr);
  }
};

class DoubleImm : public Expr {
public:
  typedef expr_nodes::DoubleImmNode Node;

  DoubleImm() = default;
  DoubleImm(const Node* n) : Expr(n) {}
  DoubleImm(double val) : DoubleImm(new Node(val)) {}

  double getVal() const { return getPtr()->val; }

private:
  const Node* getPtr() const {
    return static_cast<const Node*>(ptr);
  }
};

Add operator+(const Expr&, const Expr&);
Sub operator-(const Expr&, const Expr&);
Mul operator*(const Expr&, const Expr&);
Div operator/(const Expr&, const Expr&);

}
#endif
