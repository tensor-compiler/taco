#ifndef TACO_EXPR_NODES_H
#define TACO_EXPR_NODES_H

#include <vector>

#include "var.h"
#include "expr.h"
#include "internal_tensor.h"
#include "strings.h"
#include "expr_visitor.h"

namespace taco {

struct Add;
struct Sub;
struct Mul;
struct Div;

namespace internal {

struct Read : public TENode {
  Read(internal::Tensor tensor, const std::vector<Var>& indices) :
      tensor(tensor), indexVars(indices) {}

  void accept(internal::ExprVisitor* v) const {
    v->visit(this);
  }

  virtual void print(std::ostream& os) const {
    os << tensor.getName() << "(" << util::join(indexVars) << ")";
  }

  internal::Tensor tensor;
  std::vector<Var> indexVars;
};


struct UnaryExpr : public TENode {
  void printUnary(std::ostream& os, const std::string& op, bool prefix) const {
    if (prefix) {
      os << op;
    }
    os << operand;
    if (!prefix) {
      os << op;
    }
  }

  Expr operand;

protected:
  UnaryExpr(Expr operand) : operand(operand) {}
};


struct BinaryExpr : public TENode {
  // Syntactic sugar for arithmetic operations.
  friend Add operator+(const Expr&, const Expr&);
  friend Mul operator*(const Expr&, const Expr&);
  friend Sub operator-(const Expr&, const Expr&);
  friend Div operator/(const Expr&, const Expr&);

  void printBinary(std::ostream& os, const std::string& op) const {
    os << lhs << op << rhs;
  }

  Expr lhs;
  Expr rhs;

protected:
  BinaryExpr(Expr lhs, Expr rhs) : lhs(lhs), rhs(rhs) {}
};


struct Neg : public UnaryExpr {
  Neg(Expr operand) : UnaryExpr(operand) {}

  void accept(internal::ExprVisitor* v) const {
    v->visit(this);
  }

  void print(std::ostream& os) const {
    printUnary(os, "-", true);
  }
};


struct Add : public BinaryExpr {
  Add(Expr lhs, Expr rhs) : BinaryExpr(lhs, rhs) {}

  void accept(internal::ExprVisitor* v) const {
    v->visit(this);
  }

  void print(std::ostream& os) const {
    printBinary(os, " + ");
  }
};

struct Sub : public BinaryExpr {
  Sub(Expr lhs, Expr rhs) : BinaryExpr(lhs, rhs) {}

  void accept(internal::ExprVisitor* v) const {
    v->visit(this);
  }

  void print(std::ostream& os) const {
    printBinary(os, " - ");
  }
};

struct Mul : public BinaryExpr {
  Mul(Expr lhs, Expr rhs) : BinaryExpr(lhs, rhs) {}

  void accept(internal::ExprVisitor* v) const {
    v->visit(this);
  }

  void print(std::ostream& os) const {
    printBinary(os, " * ");
  }
};

struct Div : public BinaryExpr {
  Div(Expr lhs, Expr rhs) : BinaryExpr(lhs, rhs) {}

  void accept(internal::ExprVisitor* v) const {
    v->visit(this);
  }

  void print(std::ostream& os) const {
    printBinary(os, " / ");
  }
};

struct IntImm : public TENode {
  IntImm(int val) : val(val) {}

  void accept(internal::ExprVisitor* v) const {
    v->visit(this);
  }

  void print(std::ostream& os) const {
    os << val;
  }

  int val;
};

struct FloatImm : public TENode {
  FloatImm(float val) : val(val) {}

  void accept(internal::ExprVisitor* v) const {
    v->visit(this);
  }

  void print(std::ostream& os) const {
    os << val;
  }

  float val;
};

struct DoubleImm : public TENode {
  DoubleImm(double val) : val(val) {}

  void accept(internal::ExprVisitor* v) const {
    v->visit(this);
  }

  void print(std::ostream& os) const {
    os << val;
  }

  double val;
};


// Utility functions

/// Return the operands of the expression, in the order they appear in a
/// traversal of the expression tree.
std::vector<Tensor> getOperands(Expr);

}}
#endif
