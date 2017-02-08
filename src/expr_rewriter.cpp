#include "expr_rewriter.h"

#include "expr_nodes.h"
#include "util/collections.h"

namespace taco {
namespace internal {

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

void ExprRewriter::visit(const Read* op) {
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

void ExprRewriter::visit(const Neg* op) {
  expr = visitUnaryOp(op, this);
}

void ExprRewriter::visit(const Sqrt* op) {
  expr = visitUnaryOp(op, this);
}

void ExprRewriter::visit(const Add* op) {
  expr = visitBinaryOp(op, this);
}

void ExprRewriter::visit(const Sub* op) {
  expr = visitBinaryOp(op, this);
}

void ExprRewriter::visit(const Mul* op) {
  expr = visitBinaryOp(op, this);
}

void ExprRewriter::visit(const Div* op) {
  expr = visitBinaryOp(op, this);
}

void ExprRewriter::visit(const IntImm* op) {
  expr = op;
}

void ExprRewriter::visit(const FloatImm* op) {
  expr = op;
}

void ExprRewriter::visit(const DoubleImm* op) {
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

    void visit(const Read* op) {
      SUBSTITUTE;
    }

    void visit(const Neg* op) {
      SUBSTITUTE;
    }

    void visit(const Sqrt* op) {
      SUBSTITUTE;
    }

    void visit(const Add* op) {
      SUBSTITUTE;
    }

    void visit(const Sub* op) {
      SUBSTITUTE;
    }

    void visit(const Mul* op) {
      SUBSTITUTE;
    }

    void visit(const Div* op) {
      SUBSTITUTE;
    }

    void visit(const IntImm* op) {
      SUBSTITUTE;
    }

    void visit(const FloatImm* op) {
      SUBSTITUTE;
    }

    void visit(const DoubleImm* op) {
      SUBSTITUTE;
    }
  };

  return ReplaceRewriter(substitutions).rewrite(expr);
}

}}
