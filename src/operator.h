#ifndef TACO_OPERATOR_H
#define TACO_OPERATOR_H

#include <iostream>
#include <string>

#include "expr.h"
#include "expr_nodes.h"
#include "util/error.h"
#include "util/strings.h"

// TODO: delete
#include "expr_visitor.h"

namespace taco {
class TensorBase;
class Var;

// TODO: Make Read (,Neg,...) to a class and rename Read to Access
struct Read : public Expr {
  typedef internal::Read Node;

  Read() = default;
  Read(const Node* n);
  Read(const TensorBase& tensor);
  Read(const TensorBase& tensor, const std::vector<Var>& indices);

  const TensorBase &getTensor() const;

  const std::vector<Var>& getIndexVars() const;

  void operator=(const Expr& source) {
    assign(source);
  }
  
  void operator=(const Read& source) {
    assign(source);
  }

private:
  const Node* getPtr() const;

  void assign(Expr);
};

struct UnaryExpr : public Expr {
  typedef internal::UnaryExpr Node;
  
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

struct BinaryExpr : public Expr {
  typedef internal::BinaryExpr Node;

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

struct Neg : public UnaryExpr {
  typedef internal::Neg Node;

  Neg() = default;
  Neg(const Node* n) : UnaryExpr(n) {}
  Neg(Expr operand) : Neg(new Node(operand)) {}
};

struct Sqrt : public UnaryExpr {
  typedef internal::Sqrt Node;

  Sqrt() = default;
  Sqrt(const Node* n) : UnaryExpr(n) {}
  Sqrt(Expr operand) : Sqrt(new Node(operand)) {}
};

struct Add : public BinaryExpr {
  typedef internal::Add Node;

  Add() = default;
  Add(const Node* n) : BinaryExpr(n) {}
  Add(Expr lhs, Expr rhs) : Add(new Node(lhs, rhs)) {}
};

struct Sub : public BinaryExpr {
  typedef internal::Sub Node;

  Sub() = default;
  Sub(const Node* n) : BinaryExpr(n) {}
  Sub(Expr lhs, Expr rhs) : Sub(new Node(lhs, rhs)) {}
};

struct Mul : public BinaryExpr {
  typedef internal::Mul Node;

  Mul() = default;
  Mul(const Node* n) : BinaryExpr(n) {}
  Mul(Expr lhs, Expr rhs) : Mul(new Node(lhs, rhs)) {}
};

struct Div : public BinaryExpr {
  typedef internal::Div Node;

  Div() = default;
  Div(const Node* n) : BinaryExpr(n) {}
  Div(Expr lhs, Expr rhs) : Div(new Node(lhs, rhs)) {}
};

struct IntImm : public Expr {
  typedef internal::IntImm Node;

  IntImm() = default;
  IntImm(const Node* n) : Expr(n) {}
  IntImm(int val) : IntImm(new Node(val)) {}

  int getVal() const { return getPtr()->val; }

private:
  const Node* getPtr() const {
    return static_cast<const Node*>(ptr);
  }
};

struct FloatImm : public Expr {
  typedef internal::FloatImm Node;

  FloatImm() = default;
  FloatImm(const Node* n) : Expr(n) {}
  FloatImm(float val) : FloatImm(new Node(val)) {}

  float getVal() const { return getPtr()->val; }

private:
  const Node* getPtr() const {
    return static_cast<const Node*>(ptr);
  }
};

struct DoubleImm : public Expr {
  typedef internal::DoubleImm Node;

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
