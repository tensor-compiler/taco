#include "taco/expr_nodes/expr_rewriter.h"

#include "taco/expr_nodes/expr_nodes.h"
#include "taco/util/collections.h"

namespace taco {
namespace expr_nodes {

// class ExprRewriterStrict
IndexExpr ExprRewriterStrict::rewrite(IndexExpr e) {
  if (e.defined()) {
    e.accept(this);
    e = expr;
  }
  else {
    e = IndexExpr();
  }
  expr = IndexExpr();
  return e;
}

// class ExprRewriter
void ExprRewriter::visit(const ReadNode* op) {
  expr = op;
}

template <class T>
IndexExpr visitUnaryOp(const T *op, ExprRewriter *rw) {
  IndexExpr a = rw->rewrite(op->a);
  if (a == op->a) {
    return op;
  }
  else {
    return new T(a);
  }
}

template <class T>
IndexExpr visitBinaryOp(const T *op, ExprRewriter *rw) {
  IndexExpr a = rw->rewrite(op->a);
  IndexExpr b = rw->rewrite(op->b);
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
  IndexExpr e = op;                             \
  if (util::contains(substitutions, e)) {  \
    expr = substitutions.at(e);            \
  }                                        \
  else {                                   \
    ExprRewriter::visit(op);               \
  }                                        \
} while(false)

IndexExpr replace(IndexExpr expr,
                  const std::map<IndexExpr,IndexExpr>& substitutions) {
  struct ReplaceRewriter : public ExprRewriter {
    const std::map<IndexExpr,IndexExpr>& substitutions;
    ReplaceRewriter(const std::map<IndexExpr,IndexExpr>& substitutions)
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
