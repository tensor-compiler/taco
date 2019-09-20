#ifndef TACO_IR_VISITOR_H
#define TACO_IR_VISITOR_H

namespace taco {
namespace ir {
struct Literal;
struct Var;
struct Neg;
struct Sqrt;
struct Add;
struct Sub;
struct Mul;
struct Div;
struct Rem;
struct Min;
struct Max;
struct BitAnd;
struct BitOr;
struct Eq;
struct Neq;
struct Gt;
struct Lt;
struct Gte;
struct Lte;
struct And;
struct Or;
struct Cast;
struct Call;
struct IfThenElse;
struct Case;
struct Switch;
struct Load;
struct Malloc;
struct Sizeof;
struct Store;
struct For;
struct While;
struct Block;
struct Scope;
struct Function;
struct VarDecl;
struct Assign;
struct Yield;
struct Allocate;
struct Free;
struct Comment;
struct BlankLine;
struct Print;
struct GetProperty;

/// Extend this class to visit every node in the IR.
class IRVisitorStrict {
public:
  virtual ~IRVisitorStrict();
  virtual void visit(const Literal*) = 0;
  virtual void visit(const Var*) = 0;
  virtual void visit(const Neg*) = 0;
  virtual void visit(const Sqrt*) = 0;
  virtual void visit(const Add*) = 0;
  virtual void visit(const Sub*) = 0;
  virtual void visit(const Mul*) = 0;
  virtual void visit(const Div*) = 0;
  virtual void visit(const Rem*) = 0;
  virtual void visit(const Min*) = 0;
  virtual void visit(const Max*) = 0;
  virtual void visit(const BitAnd*) = 0;
  virtual void visit(const BitOr*) = 0;
  virtual void visit(const Eq*) = 0;
  virtual void visit(const Neq*) = 0;
  virtual void visit(const Gt*) = 0;
  virtual void visit(const Lt*) = 0;
  virtual void visit(const Gte*) = 0;
  virtual void visit(const Lte*) = 0;
  virtual void visit(const And*) = 0;
  virtual void visit(const Or*) = 0;
  virtual void visit(const Cast*) = 0;
  virtual void visit(const Call*) = 0;
  virtual void visit(const IfThenElse*) = 0;
  virtual void visit(const Case*) = 0;
  virtual void visit(const Switch*) = 0;
  virtual void visit(const Load*) = 0;
  virtual void visit(const Malloc*) = 0;
  virtual void visit(const Sizeof*) = 0;
  virtual void visit(const Store*) = 0;
  virtual void visit(const For*) = 0;
  virtual void visit(const While*) = 0;
  virtual void visit(const Block*) = 0;
  virtual void visit(const Scope*) = 0;
  virtual void visit(const Function*) = 0;
  virtual void visit(const VarDecl*) = 0;
  virtual void visit(const Assign*) = 0;
  virtual void visit(const Yield*) = 0;
  virtual void visit(const Allocate*) = 0;
  virtual void visit(const Free*) = 0;
  virtual void visit(const Comment*) = 0;
  virtual void visit(const BlankLine*) = 0;
  virtual void visit(const Print*) = 0;
  virtual void visit(const GetProperty*) = 0;
};


/// Extend this class to visit some nodes in the IR.
class IRVisitor : public IRVisitorStrict {
public:
  virtual ~IRVisitor();
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
