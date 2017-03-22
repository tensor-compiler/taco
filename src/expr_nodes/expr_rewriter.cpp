#include "taco/expr_nodes/expr_rewriter.h"

#include "taco/expr_nodes/expr_nodes.h"
#include "taco/util/collections.h"

namespace taco {
namespace expr_nodes {

Expr ExprRewriter::rewrite(Expr e) {
  if (e.defined()) {
    e.accept(this);
    e = expr;
  }
  else {
    e = Expr();
  }
  expr = Expr();
  return e;
}

void ExprRewriter::visit(const ReadNode* op) {
  expr = op;
}

template <class T>
Expr visitUnaryOp(const T *op, ExprRewriter *rw) {
  Expr a = rw->rewrite(op->a);
  if (a == op->a) {
    return op;
  }
  else {
    return new T(a);
  }
}

template <class T>
Expr visitBinaryOp(const T *op, ExprRewriter *rw) {
  Expr a = rw->rewrite(op->a);
  Expr b = rw->rewrite(op->b);
  if (a == op->a && b == op->b) {
    return op;
  }
  else {
    return new T(a, b);
  }
}

void ExprRewriter::visit(const NegNode* op) {
  expr = visitUnaryOp(op, this);
}

void ExprRewriter::visit(const SqrtNode* op) {
  expr = visitUnaryOp(op, this);
}

void ExprRewriter::visit(const AddNode* op) {
  expr = visitBinaryOp(op, this);
}

void ExprRewriter::visit(const SubNode* op) {
  expr = visitBinaryOp(op, this);
}

void ExprRewriter::visit(const MulNode* op) {
  expr = visitBinaryOp(op, this);
}

void ExprRewriter::visit(const DivNode* op) {
  expr = visitBinaryOp(op, this);
}

void ExprRewriter::visit(const IntImmNode* op) {
  expr = op;
}

void ExprRewriter::visit(const FloatImmNode* op) {
  expr = op;
}

void ExprRewriter::visit(const DoubleImmNode* op) {
  expr = op;
}


// Functions
#define SUBSTITUTE                         \
do {                                       \
  Expr e = op;                             \
  if (util::contains(substitutions, e)) {  \
    expr = substitutions.at(e);            \
  }                                        \
  else {                                   \
    ExprRewriter::visit(op);               \
  }                                        \
} while(false)

Expr replace(Expr expr, const std::map<Expr,Expr>& substitutions) {
  struct ReplaceRewriter : public ExprRewriter {
    const std::map<Expr,Expr>& substitutions;
    ReplaceRewriter(const std::map<Expr,Expr>& substitutions)
        : substitutions(substitutions) {}

    void visit(const ReadNode* op) {
      SUBSTITUTE;
    }

    void visit(const NegNode* op) {
      SUBSTITUTE;
    }

    void visit(const SqrtNode* op) {
      SUBSTITUTE;
    }

    void visit(const AddNode* op) {
      SUBSTITUTE;
    }

    void visit(const SubNode* op) {
      SUBSTITUTE;
    }

    void visit(const MulNode* op) {
      SUBSTITUTE;
    }

    void visit(const DivNode* op) {
      SUBSTITUTE;
    }

    void visit(const IntImmNode* op) {
      SUBSTITUTE;
    }

    void visit(const FloatImmNode* op) {
      SUBSTITUTE;
    }

    void visit(const DoubleImmNode* op) {
      SUBSTITUTE;
    }
  };

  return ReplaceRewriter(substitutions).rewrite(expr);
}

}}
