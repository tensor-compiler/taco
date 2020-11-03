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


  struct VarNode : public LinalgExprNode {
    VarNode(TensorVar tensorVar)
      : LinalgExprNode(tensorVar.getType().getDataType()), tensorVar(tensorVar) {}

    void accept(LinalgExprVisitorStrict* v) const override {
      v->visit(this);
    }

    virtual void setAssignment(const Assignment& assignment) {}

    TensorVar tensorVar;
  };

  struct LiteralNode : public LinalgExprNode {
    template <typename T> LiteralNode(T val) : LinalgExprNode(type<T>()) {
      this->val = malloc(sizeof(T));
      *static_cast<T*>(this->val) = val;
    }

    ~LiteralNode() {
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


  struct UnaryExprNode : public LinalgExprNode {
    LinalgExpr a;

  protected:
    UnaryExprNode(LinalgExpr a) : LinalgExprNode(a.getDataType()), a(a) {}
  };


  struct NegNode : public UnaryExprNode {
    NegNode(LinalgExpr operand) : UnaryExprNode(operand) {}

    void accept(LinalgExprVisitorStrict* v) const override{
      v->visit(this);
    }
  };

  struct TransposeNode : public UnaryExprNode {
    TransposeNode(LinalgExpr operand) : UnaryExprNode(operand) {}

    void accept (LinalgExprVisitorStrict* v) const override{
      v->visit(this);
    }
  };

  struct BinaryExprNode : public LinalgExprNode {
    virtual std::string getOperatorString() const = 0;

    LinalgExpr a;
    LinalgExpr b;

  protected:
    BinaryExprNode() : LinalgExprNode() {}
    BinaryExprNode(LinalgExpr a, LinalgExpr b)
      : LinalgExprNode(max_type(a.getDataType(), b.getDataType())), a(a), b(b) {}
  };


  struct AddNode : public BinaryExprNode {
    AddNode() : BinaryExprNode() {}
    AddNode(LinalgExpr a, LinalgExpr b) : BinaryExprNode(a, b) {}

    std::string getOperatorString() const override{
      return "+";
    }

    void accept(LinalgExprVisitorStrict* v) const override{
      v->visit(this);
    }
  };


  struct SubNode : public BinaryExprNode {
    SubNode() : BinaryExprNode() {}
    SubNode(LinalgExpr a, LinalgExpr b) : BinaryExprNode(a, b) {}

    std::string getOperatorString() const override{
      return "-";
    }

    void accept(LinalgExprVisitorStrict* v) const override{
      v->visit(this);
    }
  };


  struct MatMulNode : public BinaryExprNode {
    MatMulNode() : BinaryExprNode() {}
    MatMulNode(LinalgExpr a, LinalgExpr b) : BinaryExprNode(a, b) {}

    std::string getOperatorString() const override{
      return "*";
    }

    void accept(LinalgExprVisitorStrict* v) const override{
      v->visit(this);
    }
  };

struct ElemMulNode : public BinaryExprNode {
  ElemMulNode() : BinaryExprNode() {}
  ElemMulNode(LinalgExpr a, LinalgExpr b) : BinaryExprNode(a, b) {}

  std::string getOperatorString() const override{
    return "elemMul";
  }

  void accept(LinalgExprVisitorStrict* v) const override{
    v->visit(this);
  }
};

  struct DivNode : public BinaryExprNode {
    DivNode() : BinaryExprNode() {}
    DivNode(LinalgExpr a, LinalgExpr b) : BinaryExprNode(a, b) {}

    std::string getOperatorString() const override{
      return "/";
    }

    void accept(LinalgExprVisitorStrict* v) const override{
      v->visit(this);
    }
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

}
#endif //TACO_LINALG_NOTATION_NODES_H
