#include "taco/index_notation/expr_rewriter.h"

#include "taco/index_notation/expr_nodes.h"
#include "taco/util/collections.h"

namespace taco {

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
void ExprRewriter::visit(const AccessNode* op) {
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

void ExprRewriter::visit(const ComplexImmNode* op) {
  expr = op;
}

void ExprRewriter::visit(const UIntImmNode* op) {
  expr = op;
}

void ExprRewriter::visit(const ReductionNode* op) {
  IndexExpr a = rewrite(op->a);
  if (a == op->a) {
    expr = op;
  }
  else {
    expr = new ReductionNode(op->op, op->var, a);
  }
}

void ExprRewriter::visit(const AssignmentNode* op) {
  // A design decission is to not visit the rhs access expressions or the op,
  // as these are considered part of the assignment.  When visiting access
  // expressions, therefore, we only visit read access expressions.
  IndexExpr rhs = rewrite(op->rhs);
  if (rhs == op->rhs) {
    texpr = op;
  }
  else {
    texpr = new AssignmentNode(op->lhs, rhs, op->op);
  }
}


// Functions
#define SUBSTITUTE                         \
do {                                       \
  IndexExpr e = op;                        \
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

    void visit(const AccessNode* op) {
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

    void visit(const ComplexImmNode* op) {
      SUBSTITUTE;
    }

    void visit(const UIntImmNode* op) {
      SUBSTITUTE;
    }

    void visit(const ReductionNode* op) {
      SUBSTITUTE;
    }
  };

  return ReplaceRewriter(substitutions).rewrite(expr);
}

}
