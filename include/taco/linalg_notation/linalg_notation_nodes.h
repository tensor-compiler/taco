#ifndef TACO_LINALG_NOTATION_NODES_H
#define TACO_LINALG_NOTATION_NODES_H

#include <vector>
#include <memory>

#include "taco/type.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_nodes_abstract.h"
#include "taco/index_notation/index_notation_visitor.h"
#include "taco/index_notation/intrinsic.h"
#include "taco/util/strings.h"
#include "taco/linalg_notation/linalg_notation.h"
#include "taco/linalg_notation/linalg_notation_nodes_abstract.h"
#include "taco/linalg_notation/linalg_notation_visitor.h"

namespace taco {


  struct LinalgVarNode : public LinalgExprNode {
    LinalgVarNode(TensorVar tensorVar)
      : LinalgExprNode(tensorVar.getType().getDataType()), tensorVar(tensorVar) {}

    void accept(LinalgExprVisitorStrict* v) const override {
      v->visit(this);
    }

    virtual void setAssignment(const LinalgAssignment& assignment) {}

    TensorVar tensorVar;
  };

  struct LinalgLiteralNode : public LinalgExprNode {
    template <typename T> LinalgLiteralNode(T val) : LinalgExprNode(type<T>()) {
      this->val = malloc(sizeof(T));
      *static_cast<T*>(this->val) = val;
    }

    ~LinalgLiteralNode() {
      free(val);
    }

    void accept(LinalgExprVisitorStrict* v) const override {
      v->visit(this);
    }

    template <typename T> T getVal() const {
      taco_iassert(getDataType() == type<T>())
        << "Attempting to get data of wrong type";
      return *static_cast<T*>(val);
    }

    void* val;
  };


  struct LinalgUnaryExprNode : public LinalgExprNode {
    LinalgExpr a;

  protected:
    LinalgUnaryExprNode(LinalgExpr a) : LinalgExprNode(a.getDataType()), a(a) {}
  };


  struct LinalgNegNode : public LinalgUnaryExprNode {
    LinalgNegNode(LinalgExpr operand) : LinalgUnaryExprNode(operand) {}

    void accept(LinalgExprVisitorStrict* v) const override{
      v->visit(this);
    }
  };

  struct LinalgTransposeNode : public LinalgUnaryExprNode {
    LinalgTransposeNode(LinalgExpr operand) : LinalgUnaryExprNode(operand) {}

    void accept (LinalgExprVisitorStrict* v) const override{
      v->visit(this);
    }
  };

  struct LinalgBinaryExprNode : public LinalgExprNode {
    virtual std::string getOperatorString() const = 0;

    LinalgExpr a;
    LinalgExpr b;

  protected:
    LinalgBinaryExprNode() : LinalgExprNode() {}
    LinalgBinaryExprNode(LinalgExpr a, LinalgExpr b)
      : LinalgExprNode(max_type(a.getDataType(), b.getDataType())), a(a), b(b) {}
  };


  struct LinalgAddNode : public LinalgBinaryExprNode {
    LinalgAddNode() : LinalgBinaryExprNode() {}
    LinalgAddNode(LinalgExpr a, LinalgExpr b) : LinalgBinaryExprNode(a, b) {}

    std::string getOperatorString() const override{
      return "+";
    }

    void accept(LinalgExprVisitorStrict* v) const override{
      v->visit(this);
    }
  };


  struct LinalgSubNode : public LinalgBinaryExprNode {
    LinalgSubNode() : LinalgBinaryExprNode() {}
    LinalgSubNode(LinalgExpr a, LinalgExpr b) : LinalgBinaryExprNode(a, b) {}

    std::string getOperatorString() const override{
      return "-";
    }

    void accept(LinalgExprVisitorStrict* v) const override{
      v->visit(this);
    }
  };


  struct LinalgMatMulNode : public LinalgBinaryExprNode {
    LinalgMatMulNode() : LinalgBinaryExprNode() {}
    LinalgMatMulNode(LinalgExpr a, LinalgExpr b) : LinalgBinaryExprNode(a, b) {}

    std::string getOperatorString() const override{
      return "*";
    }

    void accept(LinalgExprVisitorStrict* v) const override{
      v->visit(this);
    }
  };

struct LinalgElemMulNode : public LinalgBinaryExprNode {
  LinalgElemMulNode() : LinalgBinaryExprNode() {}
  LinalgElemMulNode(LinalgExpr a, LinalgExpr b) : LinalgBinaryExprNode(a, b) {}

  std::string getOperatorString() const override{
    return "elemMul";
  }

  void accept(LinalgExprVisitorStrict* v) const override{
    v->visit(this);
  }
};

struct LinalgDivNode : public LinalgBinaryExprNode {
  LinalgDivNode() : LinalgBinaryExprNode() {}
  LinalgDivNode(LinalgExpr a, LinalgExpr b) : LinalgBinaryExprNode(a, b) {}

  std::string getOperatorString() const override{
    return "/";
  }

  void accept(LinalgExprVisitorStrict* v) const override{
    v->visit(this);
  }
};

// Linalg Statements
struct LinalgAssignmentNode : public LinalgStmtNode {
  LinalgAssignmentNode(const TensorVar& lhs, const LinalgExpr& rhs)
    : lhs(lhs), rhs(rhs) {}

  void accept(LinalgStmtVisitorStrict* v) const {
    v->visit(this);
  }

  TensorVar  lhs;
  LinalgExpr rhs;
};

/// Returns true if expression e is of type E.
template <typename E>
inline bool isa(const LinalgExprNode* e) {
  return e != nullptr && dynamic_cast<const E*>(e) != nullptr;
}

/// Casts the expression e to type E.
template <typename E>
inline const E* to(const LinalgExprNode* e) {
  taco_iassert(isa<E>(e)) <<
                          "Cannot convert " << typeid(e).name() << " to " << typeid(E).name();
  return static_cast<const E*>(e);
}

/// Returns true if statement e is of type S.
template <typename S>
inline bool isa(const LinalgStmtNode* s) {
  return s != nullptr && dynamic_cast<const S*>(s) != nullptr;
}

//template <typename I>
//inline const typename I::Node* getNode(const I& stmt) {
//  taco_iassert(isa<typename I::Node>(stmt.ptr));
//  return static_cast<const typename I::Node*>(stmt.ptr);
//}
}
#endif //TACO_LINALG_NOTATION_NODES_H
