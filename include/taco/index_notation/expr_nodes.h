#ifndef TACO_EXPR_NODES_H
#define TACO_EXPR_NODES_H

#include <vector>

#include "taco/tensor.h"
#include "taco/type.h"
#include "taco/index_notation/expr_node.h"
#include "taco/index_notation/expr_visitor.h"
#include "taco/util/strings.h"

namespace taco {

struct AccessNode : public ExprNode {
  AccessNode(TensorVar tensorVar, const std::vector<IndexVar>& indices)
      : ExprNode(tensorVar.getType().getDataType()), tensorVar(tensorVar), indexVars(indices) {}

  void accept(ExprVisitorStrict* v) const {
    v->visit(this);
  }

  /// Template method to set an index expression
  virtual void setIndexExpression(const IndexExpr& expr, bool accumulate) {
    tensorVar.setIndexExpression(indexVars, expr, accumulate);
  }


  TensorVar tensorVar;
  std::vector<IndexVar> indexVars;
};

struct ImmExprNode : public ExprNode {
  protected:
    ImmExprNode(DataType type) : ExprNode(type) {}
};

struct UnaryExprNode : public ExprNode {
  IndexExpr a;

protected:
  UnaryExprNode(IndexExpr a) : ExprNode(a.getDataType()), a(a) {}
};

struct NegNode : public UnaryExprNode {
  NegNode(IndexExpr operand) : UnaryExprNode(operand) {}

  void accept(ExprVisitorStrict* v) const {
    v->visit(this);
  }
};

struct SqrtNode : public UnaryExprNode {
  SqrtNode(IndexExpr operand) : UnaryExprNode(operand) {}

  void accept(ExprVisitorStrict* v) const {
    v->visit(this);
  }

};

struct BinaryExprNode : public ExprNode {
  virtual std::string getOperatorString() const = 0;

  IndexExpr a;
  IndexExpr b;

protected:
  BinaryExprNode() : ExprNode() {}
  BinaryExprNode(IndexExpr a, IndexExpr b)
      : ExprNode(max_type(a.getDataType(), b.getDataType())), a(a), b(b) {}
};

struct AddNode : public BinaryExprNode {
  AddNode() : BinaryExprNode() {}
  AddNode(IndexExpr a, IndexExpr b) : BinaryExprNode(a, b) {}

  std::string getOperatorString() const {
    return "+";
  }

  void accept(ExprVisitorStrict* v) const {
    v->visit(this);
  }
};

struct SubNode : public BinaryExprNode {
  SubNode(IndexExpr a, IndexExpr b) : BinaryExprNode(a, b) {}

  std::string getOperatorString() const {
    return "-";
  }

  void accept(ExprVisitorStrict* v) const {
    v->visit(this);
  }
};

struct MulNode : public BinaryExprNode {
  MulNode(IndexExpr a, IndexExpr b) : BinaryExprNode(a, b) {}

  std::string getOperatorString() const {
    return "*";
  }

  void accept(ExprVisitorStrict* v) const {
    v->visit(this);
  }
};

struct DivNode : public BinaryExprNode {
  DivNode(IndexExpr a, IndexExpr b) : BinaryExprNode(a, b) {}

  std::string getOperatorString() const {
    return "/";
  }

  void accept(ExprVisitorStrict* v) const {
    v->visit(this);
  }
};

struct ReductionNode : public ExprNode {
  ReductionNode(IndexExpr op, IndexVar var, IndexExpr a);

  void accept(ExprVisitorStrict* v) const {
     v->visit(this);
  }

  IndexExpr op;  // The binary reduction operator, which is a `BinaryExprNode`
                 // with undefined operands)
  IndexVar var;
  IndexExpr a;
};

struct IntImmNode : public ImmExprNode {
  IntImmNode(long long val) : ImmExprNode(Int(sizeof(long long)*8)), val(val) {}

  void accept(ExprVisitorStrict* v) const {
    v->visit(this);
  }

  long long val;
};

struct UIntImmNode : public ImmExprNode {
  UIntImmNode(unsigned long long val) : ImmExprNode(UInt(sizeof(long long)*8)), val(val) {}

  void accept(ExprVisitorStrict* v) const {
    v->visit(this);
  }

  unsigned long long val;
};

struct ComplexImmNode : public ImmExprNode {
  ComplexImmNode(std::complex<double> val) : ImmExprNode(Complex128), val(val){}

  void accept(ExprVisitorStrict* v) const {
    v->visit(this);
  }

  std::complex<double> val;
};

struct FloatImmNode : public ImmExprNode {
  FloatImmNode(double val) : ImmExprNode(Float()), val(val) {}

  void accept(ExprVisitorStrict* v) const {
    v->visit(this);
  }

  double val;
};

/// Returns true if expression e is of type E
// @{
template <typename E>
inline bool isa(IndexExpr e) {
  return e.defined() && dynamic_cast<const E*>(e.ptr) != nullptr;
}
template <typename E>
inline bool isa(const ExprNode* e) {
  return e != nullptr && dynamic_cast<const E*>(e) != nullptr;
}
// @}

/// Casts the expression e to type E
// @{
template <typename E>
inline const E* to(IndexExpr e) {
  taco_iassert(isa<E>(e)) <<
      "Cannot convert " << typeid(e).name() << " to " << typeid(E).name();
  return static_cast<const E*>(e.ptr);
}
template <typename E>
inline const E* to(const ExprNode* e) {
  taco_iassert(isa<E>(e)) <<
      "Cannot convert " << typeid(e).name() << " to " << typeid(E).name();
  return static_cast<const E*>(e);
}
// @}


// Utility functions

/// Returns the operands of the expression, in the ordering they appear in a
/// traversal of the expression tree.
std::vector<taco::TensorVar> getOperands(const IndexExpr&);

}
#endif
