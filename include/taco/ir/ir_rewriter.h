#ifndef TACO_IR_REWRITER_H
#define TACO_IR_REWRITER_H

#include "taco/ir/ir_visitor.h"
#include "taco/ir/ir.h"

namespace taco {
namespace ir {

/// Extend this class to rewrite IR nodes.
class IRRewriter : public IRVisitorStrict {
public:
  virtual ~IRRewriter();

  Expr rewrite(Expr);
  Stmt rewrite(Stmt);

protected:
  /// visit methods that take Exprs assign to this to return their value.
  Expr expr;

  /// visit methods that take Stmts assign to this to return their value.
  Stmt stmt;

  using IRVisitorStrict::visit;
  virtual void visit(const Literal* op);
  virtual void visit(const Var* op);
  virtual void visit(const Neg* op);
  virtual void visit(const Sqrt* op);
  virtual void visit(const Add* op);
  virtual void visit(const Sub* op);
  virtual void visit(const Mul* op);
  virtual void visit(const Div* op);
  virtual void visit(const Rem* op);
  virtual void visit(const Min* op);
  virtual void visit(const Max* op);
  virtual void visit(const BitAnd* op);
  virtual void visit(const BitOr* op);
  virtual void visit(const Eq* op);
  virtual void visit(const Neq* op);
  virtual void visit(const Gt* op);
  virtual void visit(const Lt* op);
  virtual void visit(const Gte* op);
  virtual void visit(const Lte* op);
  virtual void visit(const And* op);
  virtual void visit(const Or* op);
  virtual void visit(const Cast* op);
  virtual void visit(const Call* op);
  virtual void visit(const IfThenElse* op);
  virtual void visit(const Case* op);
  virtual void visit(const Switch* op);
  virtual void visit(const Load* op);
  virtual void visit(const Malloc* op);
  virtual void visit(const Sizeof* op);
  virtual void visit(const Store* op);
  virtual void visit(const For* op);
  virtual void visit(const While* op);
  virtual void visit(const Block* op);
  virtual void visit(const Scope* op);
  virtual void visit(const Function* op);
  virtual void visit(const VarDecl* op);
  virtual void visit(const Assign* op);
  virtual void visit(const Yield* op);
  virtual void visit(const Allocate* op);
  virtual void visit(const Free* op);
  virtual void visit(const Comment* op);
  virtual void visit(const BlankLine* op);
  virtual void visit(const Print* op);
  virtual void visit(const GetProperty* op);
};

}}
#endif
