#ifndef TACO_EXPR_NODES_H
#define TACO_EXPR_NODES_H

#include <vector>

#include "taco/tensor_base.h"
#include "taco/expr.h"
#include "taco/expr_nodes/expr_visitor.h"
#include "taco/util/strings.h"

namespace taco {
namespace expr_nodes {

struct ReadNode : public ExprNode {
  ReadNode(TensorBase tensor, const std::vector<Var>& indices) :
      tensor(tensor), indexVars(indices) {}

  void accept(ExprVisitorStrict* v) const {
    v->visit(this);
  }

  virtual void print(std::ostream& os) const {
    os << tensor.getName() << "(" << util::join(indexVars) << ")";
  }

  TensorBase tensor;
  std::vector<Var> indexVars;
};

struct ImmExprNode : public ExprNode {
};

struct UnaryExprNode : public ExprNode {
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
  UnaryExprNode(Expr a) : a(a) {}
};

struct BinaryExprNode : public ExprNode {
  void printBinary(std::ostream& os, const std::string& op) const {
    os << "(" << a << op << b << ")";
  }

  Expr a;
  Expr b;

protected:
  BinaryExprNode(Expr a, Expr b) : a(a), b(b) {}
};

struct NegNode : public UnaryExprNode {
  NegNode(Expr operand) : UnaryExprNode(operand) {}

  void accept(ExprVisitorStrict* v) const {
    v->visit(this);
  }

  void print(std::ostream& os) const {
    printUnary(os, "-", true);
  }
};

struct SqrtNode : public UnaryExprNode {
  SqrtNode(Expr operand) : UnaryExprNode(operand) {}

  void accept(ExprVisitorStrict* v) const {
    v->visit(this);
  }

  void print(std::ostream& os) const {
    printUnary(os, "sqrt", true);
  }
};

struct AddNode : public BinaryExprNode {
  AddNode(Expr a, Expr b) : BinaryExprNode(a, b) {}

  void accept(ExprVisitorStrict* v) const {
    v->visit(this);
  }

  void print(std::ostream& os) const {
    printBinary(os, " + ");
  }
};

struct SubNode : public BinaryExprNode {
  SubNode(Expr a, Expr b) : BinaryExprNode(a, b) {}

  void accept(ExprVisitorStrict* v) const {
    v->visit(this);
  }

  void print(std::ostream& os) const {
    printBinary(os, " - ");
  }
};

struct MulNode : public BinaryExprNode {
  MulNode(Expr a, Expr b) : BinaryExprNode(a, b) {}

  void accept(ExprVisitorStrict* v) const {
    v->visit(this);
  }

  void print(std::ostream& os) const {
    printBinary(os, " * ");
  }
};

struct DivNode : public BinaryExprNode {
  DivNode(Expr a, Expr b) : BinaryExprNode(a, b) {}

  void accept(ExprVisitorStrict* v) const {
    v->visit(this);
  }

  void print(std::ostream& os) const {
    printBinary(os, " / ");
  }
};

struct IntImmNode : public ImmExprNode {
  IntImmNode(int val) : val(val) {}

  void accept(ExprVisitorStrict* v) const {
    v->visit(this);
  }

  void print(std::ostream& os) const {
    os << val;
  }

  int val;
};

struct FloatImmNode : public ImmExprNode {
  FloatImmNode(float val) : val(val) {}

  void accept(ExprVisitorStrict* v) const {
    v->visit(this);
  }

  void print(std::ostream& os) const {
    os << val;
  }

  float val;
};

struct DoubleImmNode : public ImmExprNode {
  DoubleImmNode(double val) : val(val) {}

  void accept(ExprVisitorStrict* v) const {
    v->visit(this);
  }

  void print(std::ostream& os) const {
    os << val;
  }

  double val;
};

/// Returns true if expression e is of type E
// @{
template <typename E>
inline bool isa(Expr e) {
  return e.defined() && dynamic_cast<const E*>(e.ptr) != nullptr;
}
template <typename E>
inline bool isa(const expr_nodes::ExprNode* e) {
  return e != nullptr && dynamic_cast<const E*>(e) != nullptr;
}
// @{

/// Casts the expression e to type E
// @{
template <typename E>
inline const E* to(Expr e) {
  taco_iassert(isa<E>(e)) <<
      "Cannot convert " << typeid(e).name() << " to " <<typeid(E).name();
  return static_cast<const E*>(e.ptr);
}
template <typename E>
inline const E* to(const expr_nodes::ExprNode* e) {
  taco_iassert(isa<E>(e)) <<
      "Cannot convert " << typeid(e).name() << " to " <<typeid(E).name();
  return static_cast<const E*>(e);
}
// @{

// Utility functions

/// Return the operands of the expression, in the order they appear in a
/// traversal of the expression tree.
std::vector<taco::TensorBase> getOperands(const taco::Expr&);

}}
#endif
