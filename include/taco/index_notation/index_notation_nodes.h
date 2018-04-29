#ifndef TACO_INDEX_NOTATION_NODES_H
#define TACO_INDEX_NOTATION_NODES_H

#include <vector>

#include "taco/tensor.h"
#include "taco/type.h"
#include "taco/index_notation/index_notation_nodes_abstract.h"
#include "taco/index_notation/index_notation_visitor.h"
#include "taco/util/strings.h"

namespace taco {

// Scalar Index Expressions

struct AccessNode : public IndexExprNode {
  AccessNode(TensorVar tensorVar, const std::vector<IndexVar>& indices)
      : IndexExprNode(tensorVar.getType().getDataType()), tensorVar(tensorVar), indexVars(indices) {}

  void accept(IndexExprVisitorStrict* v) const {
    v->visit(this);
  }

  virtual void setAssignment(const Assignment& assignment) {
    tensorVar.setAssignment(assignment);
  }

  TensorVar tensorVar;
  std::vector<IndexVar> indexVars;
};

struct ImmExprNode : public IndexExprNode {
  protected:
    ImmExprNode(DataType type) : IndexExprNode(type) {}
};

struct UnaryExprNode : public IndexExprNode {
  IndexExpr a;

protected:
  UnaryExprNode(IndexExpr a) : IndexExprNode(a.getDataType()), a(a) {}
};

struct NegNode : public UnaryExprNode {
  NegNode(IndexExpr operand) : UnaryExprNode(operand) {}

  void accept(IndexExprVisitorStrict* v) const {
    v->visit(this);
  }
};

struct SqrtNode : public UnaryExprNode {
  SqrtNode(IndexExpr operand) : UnaryExprNode(operand) {}

  void accept(IndexExprVisitorStrict* v) const {
    v->visit(this);
  }

};

struct BinaryExprNode : public IndexExprNode {
  virtual std::string getOperatorString() const = 0;

  IndexExpr a;
  IndexExpr b;

protected:
  BinaryExprNode() : IndexExprNode() {}
  BinaryExprNode(IndexExpr a, IndexExpr b)
      : IndexExprNode(max_type(a.getDataType(), b.getDataType())), a(a), b(b) {}
};

struct AddNode : public BinaryExprNode {
  AddNode() : BinaryExprNode() {}
  AddNode(IndexExpr a, IndexExpr b) : BinaryExprNode(a, b) {}

  std::string getOperatorString() const {
    return "+";
  }

  void accept(IndexExprVisitorStrict* v) const {
    v->visit(this);
  }
};

struct SubNode : public BinaryExprNode {
  SubNode(IndexExpr a, IndexExpr b) : BinaryExprNode(a, b) {}

  std::string getOperatorString() const {
    return "-";
  }

  void accept(IndexExprVisitorStrict* v) const {
    v->visit(this);
  }
};

struct MulNode : public BinaryExprNode {
  MulNode(IndexExpr a, IndexExpr b) : BinaryExprNode(a, b) {}

  std::string getOperatorString() const {
    return "*";
  }

  void accept(IndexExprVisitorStrict* v) const {
    v->visit(this);
  }
};

struct DivNode : public BinaryExprNode {
  DivNode(IndexExpr a, IndexExpr b) : BinaryExprNode(a, b) {}

  std::string getOperatorString() const {
    return "/";
  }

  void accept(IndexExprVisitorStrict* v) const {
    v->visit(this);
  }
};

struct ReductionNode : public IndexExprNode {
  ReductionNode(IndexExpr op, IndexVar var, IndexExpr a);

  void accept(IndexExprVisitorStrict* v) const {
     v->visit(this);
  }

  IndexExpr op;  // The binary reduction operator, which is a `BinaryExprNode`
                 // with undefined operands)
  IndexVar var;
  IndexExpr a;
};

struct IntImmNode : public ImmExprNode {
  IntImmNode(long long val) : ImmExprNode(Int(sizeof(long long)*8)), val(val) {}

  void accept(IndexExprVisitorStrict* v) const {
    v->visit(this);
  }

  long long val;
};

struct UIntImmNode : public ImmExprNode {
  UIntImmNode(unsigned long long val) : ImmExprNode(UInt(sizeof(long long)*8)), val(val) {}

  void accept(IndexExprVisitorStrict* v) const {
    v->visit(this);
  }

  unsigned long long val;
};

struct ComplexImmNode : public ImmExprNode {
  ComplexImmNode(std::complex<double> val) : ImmExprNode(Complex128), val(val){}

  void accept(IndexExprVisitorStrict* v) const {
    v->visit(this);
  }

  std::complex<double> val;
};

struct FloatImmNode : public ImmExprNode {
  FloatImmNode(double val) : ImmExprNode(Float()), val(val) {}

  void accept(IndexExprVisitorStrict* v) const {
    v->visit(this);
  }

  double val;
};


// Index Statements
struct AssignmentNode : public IndexStmtNode {
  AssignmentNode(const Access& lhs, const IndexExpr& rhs, const IndexExpr& op)
      : lhs(lhs), rhs(rhs), op(op) {}

  void accept(IndexNotationVisitorStrict* v) const {
    v->visit(this);
  }

  Access    lhs;
  IndexExpr rhs;
  IndexExpr op;
};

struct ForallNode : public IndexStmtNode {
  ForallNode(IndexVar indexVar, IndexStmt stmt)
      : indexVar(indexVar), stmt(stmt) {}

  void accept(IndexNotationVisitorStrict* v) const {
    v->visit(this);
  }

  IndexVar indexVar;
  IndexStmt stmt;
};

struct WhereNode : public IndexStmtNode {
  WhereNode(IndexStmt consumer, IndexStmt producer)
      : consumer(consumer), producer(producer) {}

  void accept(IndexNotationVisitorStrict* v) const {
    v->visit(this);
  }

  IndexStmt consumer;
  IndexStmt producer;
};


/// Returns true if expression e is of type E.
template <typename E>
inline bool isa(const IndexExprNode* e) {
  return e != nullptr && dynamic_cast<const E*>(e) != nullptr;
}

/// Casts the expression e to type E.
template <typename E>
inline const E* to(const IndexExprNode* e) {
  taco_iassert(isa<E>(e)) <<
      "Cannot convert " << typeid(e).name() << " to " << typeid(E).name();
  return static_cast<const E*>(e);
}

/// Returns true if expression e is of type E.
template <typename E>
inline bool isa(IndexExpr e) {
  return e.defined() && dynamic_cast<const E*>(e.ptr) != nullptr;
}

/// Casts the expression e to type E.
template <typename E>
inline const E* to(IndexExpr e) {
  taco_iassert(isa<E>(e)) <<
      "Cannot convert " << typeid(e).name() << " to " << typeid(E).name();
  return static_cast<const E*>(e.ptr);
}

/// Returns true if statement e is of type S.
template <typename S>
inline bool isa(const IndexStmtNode* s) {
  return s != nullptr && dynamic_cast<const S*>(s) != nullptr;
}

/// Casts the statement s to type S.
template <typename S>
inline const S* to(const IndexStmtNode* s) {
  taco_iassert(isa<S>(s)) <<
      "Cannot convert " << typeid(s).name() << " to " << typeid(S).name();
  return static_cast<const S*>(s);
}

/// Get the node of an assignment statement.
static inline const AssignmentNode* getNode(const Assignment& assignment) {
  taco_iassert(isa<AssignmentNode>(assignment.ptr));
  return static_cast<const AssignmentNode*>(assignment.ptr);
}

/// Get the node of a forall statement.
static inline const ForallNode* getNode(const Forall& forall) {
  taco_iassert(isa<ForallNode>(forall.ptr));
  return static_cast<const ForallNode*>(forall.ptr);
}

/// Get the node of a where statement.
static inline const WhereNode* getNode(const Where& where) {
  taco_iassert(isa<WhereNode>(where.ptr));
  return static_cast<const WhereNode*>(where.ptr);
}

/// Returns the operands of the expression, in the ordering they appear in a
/// traversal of the expression tree.
std::vector<taco::TensorVar> getOperands(const IndexExpr&);

}
#endif
