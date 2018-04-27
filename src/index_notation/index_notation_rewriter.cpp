#include "taco/index_notation/index_notation_rewriter.h"

#include "taco/index_notation/index_notation_nodes.h"
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


// class IndexNotationRewriterStrict
IndexStmt IndexNotationRewriterStrict::rewrite(IndexStmt s) {
  if (s.defined()) {
    s.accept(this);
    s = stmt;
  }
  else {
    s = IndexStmt();
  }
  stmt = IndexStmt();
  return s;
}


// class ExprRewriter
void IndexNotationRewriter::visit(const AccessNode* op) {
  expr = op;
}

template <class T>
IndexExpr visitUnaryOp(const T *op, IndexNotationRewriter *rw) {
  IndexExpr a = rw->rewrite(op->a);
  if (a == op->a) {
    return op;
  }
  else {
    return new T(a);
  }
}

template <class T>
IndexExpr visitBinaryOp(const T *op, IndexNotationRewriter *rw) {
  IndexExpr a = rw->rewrite(op->a);
  IndexExpr b = rw->rewrite(op->b);
  if (a == op->a && b == op->b) {
    return op;
  }
  else {
    return new T(a, b);
  }
}

void IndexNotationRewriter::visit(const NegNode* op) {
  expr = visitUnaryOp(op, this);
}

void IndexNotationRewriter::visit(const SqrtNode* op) {
  expr = visitUnaryOp(op, this);
}

void IndexNotationRewriter::visit(const AddNode* op) {
  expr = visitBinaryOp(op, this);
}

void IndexNotationRewriter::visit(const SubNode* op) {
  expr = visitBinaryOp(op, this);
}

void IndexNotationRewriter::visit(const MulNode* op) {
  expr = visitBinaryOp(op, this);
}

void IndexNotationRewriter::visit(const DivNode* op) {
  expr = visitBinaryOp(op, this);
}

void IndexNotationRewriter::visit(const IntImmNode* op) {
  expr = op;
}

void IndexNotationRewriter::visit(const FloatImmNode* op) {
  expr = op;
}

void IndexNotationRewriter::visit(const ComplexImmNode* op) {
  expr = op;
}

void IndexNotationRewriter::visit(const UIntImmNode* op) {
  expr = op;
}

void IndexNotationRewriter::visit(const ReductionNode* op) {
  IndexExpr a = rewrite(op->a);
  if (a == op->a) {
    expr = op;
  }
  else {
    expr = new ReductionNode(op->op, op->var, a);
  }
}

void IndexNotationRewriter::visit(const AssignmentNode* op) {
  // A design decission is to not visit the rhs access expressions or the op,
  // as these are considered part of the assignment.  When visiting access
  // expressions, therefore, we only visit read access expressions.
  IndexExpr rhs = rewrite(op->rhs);
  if (rhs == op->rhs) {
    stmt = op;
  }
  else {
    stmt = new AssignmentNode(op->lhs, rhs, op->op);
  }
}

void IndexNotationRewriter::visit(const ForallNode* op) {
  IndexStmt stmt = rewrite(op->stmt);
  if (stmt == op->stmt) {
    stmt = op;
  }
  else {
    stmt = new ForallNode(op->indexVar, stmt);
  }
}

void IndexNotationRewriter::visit(const WhereNode* op) {
  IndexStmt producer = rewrite(op->producer);
  IndexStmt consumer = rewrite(op->consumer);
  if (producer == op->producer && consumer == op->consumer) {
    stmt = op;
  }
  else {
    stmt = new WhereNode(consumer, producer);
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
    IndexNotationRewriter::visit(op);      \
  }                                        \
} while(false)

IndexExpr replace(IndexExpr expr,
                  const std::map<IndexExpr,IndexExpr>& substitutions) {
  struct ReplaceRewriter : public IndexNotationRewriter {

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
