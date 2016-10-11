#ifndef TACO_EXPR_NODES_H
#define TACO_EXPR_NODES_H

#include <vector>

#include "var.h"
#include "expr.h"
#include "internal_tensor.h"
#include "strings.h"
#include "expr_visitor.h"

namespace taco {

struct NaryExpr;
struct Add;
struct Sub;
struct Mul;
struct Div;

namespace internal {


struct ReadNode : public internal::TENode {
  ReadNode(internal::Tensor tensor, const std::vector<Var>& indices) :
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


struct NegNode : public internal::TENode {
  NegNode(Expr a) : a(a) {}

  void accept(internal::ExprVisitor* v) const {
    v->visit(this);
  }

  void print(std::ostream& os) const {
    os << "-" << a;
  }

  Expr a;
};


struct NaryExprNode : public internal::TENode {
  template <typename T>
  friend std::vector<Expr> mergeOperands(const Expr&, const Expr&);

  // Syntactic sugar for arithmetic operations.
  friend Add operator+(const Expr&, const Expr&);
  friend Mul operator*(const Expr&, const Expr&);

  void printNary(std::ostream& os, const std::string& op) const {
    os << util::join(operands, op);
  }

  std::vector<Expr> operands;

protected:
  NaryExprNode(const std::vector<Expr>& operands) : operands(operands) {}
};


struct BinaryExprNode : public internal::TENode {
  // Syntactic sugar for arithmetic operations.
  friend Sub operator-(const Expr&, const Expr&);
  friend Div operator/(const Expr&, const Expr&);

  void printBinary(std::ostream& os, const std::string& op) const {
    os << lhs << op << rhs;
  }

  Expr lhs;
  Expr rhs;

protected:
  BinaryExprNode(Expr lhs, Expr rhs) : lhs(lhs), rhs(rhs) {}
};


struct AddNode : public NaryExprNode {
  AddNode(const std::vector<Expr>& operands) : NaryExprNode(operands) {}

  void accept(internal::ExprVisitor* v) const {
    v->visit(this);
  }

  void print(std::ostream& os) const {
    printNary(os, " + ");
  }
};

struct SubNode : public BinaryExprNode {
  SubNode(Expr lhs, Expr rhs) : BinaryExprNode(lhs, rhs) {}

  void accept(internal::ExprVisitor* v) const {
    v->visit(this);
  }

  void print(std::ostream& os) const {
    printBinary(os, " - ");
  }
};

struct MulNode : public NaryExprNode {
  MulNode(const std::vector<Expr>& operands) : NaryExprNode(operands) {}

  void accept(internal::ExprVisitor* v) const {
    v->visit(this);
  }

  void print(std::ostream& os) const {
    printNary(os, " * ");
  }
};

struct DivNode : public BinaryExprNode {
  DivNode(Expr lhs, Expr rhs) : BinaryExprNode(lhs, rhs) {}

  void accept(internal::ExprVisitor* v) const {
    v->visit(this);
  }

  void print(std::ostream& os) const {
    printBinary(os, " / ");
  }
};

struct IntImmNode : public internal::TENode {
  IntImmNode(int val) : val(val) {}

  void accept(internal::ExprVisitor* v) const {
    v->visit(this);
  }

  void print(std::ostream& os) const {
    os << val;
  }

  int val;
};

struct FloatImmNode : public internal::TENode {
  FloatImmNode(float val) : val(val) {}

  void accept(internal::ExprVisitor* v) const {
    v->visit(this);
  }

  void print(std::ostream& os) const {
    os << val;
  }

  float val;
};

struct DoubleImmNode : public internal::TENode {
  DoubleImmNode(double val) : val(val) {}

  void accept(internal::ExprVisitor* v) const {
    v->visit(this);
  }

  void print(std::ostream& os) const {
    os << val;
  }

  double val;
};

}}
#endif
