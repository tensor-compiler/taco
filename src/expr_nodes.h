#ifndef TACO_EXPR_NODES_H
#define TACO_EXPR_NODES_H

#include <vector>

#include "tensor_base.h"
#include "var.h"
#include "expr.h"
#include "strings.h"
#include "expr_visitor.h"

namespace taco {

struct Add;
struct Sub;
struct Mul;
struct Div;

namespace internal {

struct Read : public TENode {
  Read(TensorBase tensor, const std::vector<Var>& indices) :
      tensor(tensor), indexVars(indices) {}

  void accept(internal::ExprVisitorStrict* v) const {
    v->visit(this);
  }

  virtual void print(std::ostream& os) const {
    os << tensor.getName() << "(" << util::join(indexVars) << ")";
  }

  TensorBase tensor;
  std::vector<Var> indexVars;
};

struct ImmExpr : public TENode {
};

struct UnaryExpr : public TENode {
  void printUnary(std::ostream& os, const std::string& op, bool prefix) const {
    if (prefix) {
      os << op;
    }
    os << "(" << a << ")";
    if (!prefix) {
      os << op;
    }
  }

  Expr a;

protected:
  UnaryExpr(Expr a) : a(a) {}
};

struct BinaryExpr : public TENode {
  // Syntactic sugar for arithmetic operations.
  friend Add operator+(const Expr&, const Expr&);
  friend Mul operator*(const Expr&, const Expr&);
  friend Sub operator-(const Expr&, const Expr&);
  friend Div operator/(const Expr&, const Expr&);

  void printBinary(std::ostream& os, const std::string& op) const {
    os << "(" << a << ")" << op << "(" << b << ")";
  }

  Expr a;
  Expr b;

protected:
  BinaryExpr(Expr a, Expr b) : a(a), b(b) {}
};

struct Neg : public UnaryExpr {
  Neg(Expr operand) : UnaryExpr(operand) {}

  void accept(internal::ExprVisitorStrict* v) const {
    v->visit(this);
  }

  void print(std::ostream& os) const {
    printUnary(os, "-", true);
  }
};

struct Sqrt : public UnaryExpr {
  Sqrt(Expr operand) : UnaryExpr(operand) {}

  void accept(internal::ExprVisitorStrict* v) const {
    v->visit(this);
  }

  void print(std::ostream& os) const {
    printUnary(os, "sqrt", true);
  }
};

struct Add : public BinaryExpr {
  Add(Expr a, Expr b) : BinaryExpr(a, b) {}

  void accept(internal::ExprVisitorStrict* v) const {
    v->visit(this);
  }

  void print(std::ostream& os) const {
    printBinary(os, " + ");
  }
};

struct Sub : public BinaryExpr {
  Sub(Expr a, Expr b) : BinaryExpr(a, b) {}

  void accept(internal::ExprVisitorStrict* v) const {
    v->visit(this);
  }

  void print(std::ostream& os) const {
    printBinary(os, " - ");
  }
};

struct Mul : public BinaryExpr {
  Mul(Expr a, Expr b) : BinaryExpr(a, b) {}

  void accept(internal::ExprVisitorStrict* v) const {
    v->visit(this);
  }

  void print(std::ostream& os) const {
    printBinary(os, " * ");
  }
};

struct Div : public BinaryExpr {
  Div(Expr a, Expr b) : BinaryExpr(a, b) {}

  void accept(internal::ExprVisitorStrict* v) const {
    v->visit(this);
  }

  void print(std::ostream& os) const {
    printBinary(os, " / ");
  }
};

struct IntImm : public ImmExpr {
  IntImm(int val) : val(val) {}

  void accept(internal::ExprVisitorStrict* v) const {
    v->visit(this);
  }

  void print(std::ostream& os) const {
    os << val;
  }

  int val;
};

struct FloatImm : public ImmExpr {
  FloatImm(float val) : val(val) {}

  void accept(internal::ExprVisitorStrict* v) const {
    v->visit(this);
  }

  void print(std::ostream& os) const {
    os << val;
  }

  float val;
};

struct DoubleImm : public ImmExpr {
  DoubleImm(double val) : val(val) {}

  void accept(internal::ExprVisitorStrict* v) const {
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
std::vector<TensorBase> getOperands(Expr);

}}
#endif
