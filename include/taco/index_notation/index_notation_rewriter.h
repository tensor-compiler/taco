#ifndef TACO_INDEX_NOTATION_REWRITER_H
#define TACO_INDEX_NOTATION_REWRITER_H

#include <map>

#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_visitor.h"

namespace taco {


/// Extend this class to rewrite all index expressions.
class ExprRewriterStrict : public IndexExprVisitorStrict {
public:
  virtual ~ExprRewriterStrict() {}

  /// Rewrite an index expression.
  IndexExpr rewrite(IndexExpr);

protected:
  using IndexExprVisitorStrict::visit;

  /// Assign to expr in visit methods to replace the visited expr.
  IndexExpr expr;
};


/// Extend this class to rewrite all index expressions and statements.
class IndexNotationRewriterStrict : public ExprRewriterStrict,
                                    public IndexNotationVisitorStrict {
public:
  virtual ~IndexNotationRewriterStrict() {}

  using ExprRewriterStrict::rewrite;

  /// Rewrite an index statement.
  IndexStmt rewrite(IndexStmt);

protected:
  using ExprRewriterStrict::visit;
  using IndexNotationVisitorStrict::visit;

  /// Assign to stmt in visit methods to replace the visited stmt.
  IndexStmt stmt;
};


/// Extend this class to rewrite some index expressions and statements.
class IndexNotationRewriter : public IndexNotationRewriterStrict {
public:
  virtual ~IndexNotationRewriter() {}

protected:
  using IndexNotationRewriterStrict::visit;

  virtual void visit(const AccessNode* op);
  virtual void visit(const NegNode* op);
  virtual void visit(const SqrtNode* op);
  virtual void visit(const AddNode* op);
  virtual void visit(const SubNode* op);
  virtual void visit(const MulNode* op);
  virtual void visit(const DivNode* op);
  virtual void visit(const IntImmNode* op);
  virtual void visit(const FloatImmNode* op);
  virtual void visit(const ComplexImmNode* op);
  virtual void visit(const UIntImmNode* op);
  virtual void visit(const ReductionNode* op);

  virtual void visit(const AssignmentNode* op);
  virtual void visit(const ForallNode* op);
  virtual void visit(const WhereNode* op);
};


/// Rewrites the expression to replace sub-expressions with new expressions.
IndexExpr replace(IndexExpr expr,
                  const std::map<IndexExpr,IndexExpr>& substitutions);

}
#endif
