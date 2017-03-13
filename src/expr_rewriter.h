#ifndef TACO_EXPR_REWRITER_H
#define TACO_EXPR_REWRITER_H

#include <map>

#include "taco/expr.h"
#include "expr_visitor.h"

namespace taco {
namespace internal {

struct Read;
struct Neg;
struct Sqrt;
struct Add;
struct Sub;
struct Mul;
struct Div;
struct IntImm;
struct FloatImm;
struct DoubleImm;

/// Inherit from this class and override methods to rewrite expressions.
class ExprRewriter : public ExprVisitorStrict {
public:
  virtual ~ExprRewriter() {}

  /// Rewrite expr using rules defined by an ExprRewriter sub-class
  Expr rewrite(Expr);

protected:
  using ExprVisitorStrict::visit;

  /// assign to expr in visit methods to replace the visited expr
  Expr expr;

  virtual void visit(const Read* op);
  virtual void visit(const Neg* op);
  virtual void visit(const Sqrt* op);
  virtual void visit(const Add* op);
  virtual void visit(const Sub* op);
  virtual void visit(const Mul* op);
  virtual void visit(const Div* op);
  virtual void visit(const IntImm* op);
  virtual void visit(const FloatImm* op);
  virtual void visit(const DoubleImm* op);
};


/// Rewrites the expression to replace sub-expressions with new expressions.
Expr replace(Expr expr, const std::map<Expr,Expr>& substitutions);

}}
#endif
