#ifndef TACO_EXPR_REWRITER_H
#define TACO_EXPR_REWRITER_H

#include <map>

#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/expr_visitor.h"

namespace taco {

struct AccessNode;
struct NegNode;
struct SqrtNode;
struct AddNode;
struct SubNode;
struct MulNode;
struct DivNode;
struct IntImmNode;
struct FloatImmNode;
struct UIntImmNode;
struct ComplexImmNode;
struct ReductionNode;

struct AssignmentNode;

class ExprRewriterStrict : public ExprVisitorStrict {
public:
  virtual ~ExprRewriterStrict() {}

  /// Rewrite an expr using rules defined by an ExprRewriter sub-class
  IndexExpr rewrite(IndexExpr);
  TensorExpr rewrite(TensorExpr);

protected:
  using ExprVisitorStrict::visit;

  /// assign to expr in visit methods to replace the visited expr
  IndexExpr expr;
  TensorExpr texpr;
};

/// Inherit from this class and override methods to rewrite expressions.
class ExprRewriter : public ExprRewriterStrict {
public:
  virtual ~ExprRewriter() {}

protected:
  using ExprRewriterStrict::visit;

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
};


/// Rewrites the expression to replace sub-expressions with new expressions.
IndexExpr replace(IndexExpr expr,
                  const std::map<IndexExpr,IndexExpr>& substitutions);

}
#endif
