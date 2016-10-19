#include "ir.h"
#include "ir_visitor.h"
#include "ir_printer.h"

namespace taco {
namespace internal {

Expr::Expr(int n) : IRHandle(Literal::make(n)) {
}

Expr Literal::make(double val, ComponentType type) {
  Literal *lit = new Literal;
  lit->type = type;
  lit->dbl_value = val;
  return lit;
}

Expr Literal::make(int val) {
  Literal *lit = new Literal;
  lit->type = typeOf<int>();
  lit->value = (int64_t)val;
  return lit;
}

Expr Var::make(std::string name, ComponentType type, bool is_ptr) {
  Var *var = new Var;
  var->type = type;
  var->name = name;
  var->is_ptr = is_ptr;
  return var;
}

// Binary Expressions
// helper
ComponentType max_type(Expr a, Expr b) {
  iassert(a.type() != typeOf<bool>()) << "Can't do arithmetic on booleans.";
  iassert(b.type() != typeOf<bool>()) << "Can't do arithmetic on booleans.";

  if (a.type() == b.type()) {
    return a.type();
  } else {
    // if either are double, make it double
    if (a.type() == typeOf<double>() || b.type() == typeOf<double>())
      return typeOf<double>();
    else
      return typeOf<float>();
  }
}

Expr Add::make(Expr a, Expr b) {
  return Add::make(a, b, max_type(a, b));
}

Expr Add::make(Expr a, Expr b, ComponentType type) {
//  iassert(a.type() != typeOf<bool>()) << "Can't do arithmetic on booleans.";
//  iassert(b.type() != typeOf<bool>()) << "Can't do arithmetic on booleans.";

  Add *add = new Add;
  add->type = type;
  add->a = a;
  add->b = b;
  return add;
}

Expr Sub::make(Expr a, Expr b) {
  return Sub::make(a, b, max_type(a, b));
}

Expr Sub::make(Expr a, Expr b, ComponentType type) {
  iassert(a.type() != typeOf<bool>()) << "Can't do arithmetic on booleans.";
  iassert(b.type() != typeOf<bool>()) << "Can't do arithmetic on booleans.";

  Sub *sub = new Sub;
  sub->type = type;
  sub->a = a;
  sub->b = b;
  return sub;
}

Expr Mul::make(Expr a, Expr b) {
  return Mul::make(a, b, max_type(a, b));
}

Expr Mul::make(Expr a, Expr b, ComponentType type) {
  iassert(a.type() != typeOf<bool>()) << "Can't do arithmetic on booleans.";
  iassert(b.type() != typeOf<bool>()) << "Can't do arithmetic on booleans.";

  Mul *mul = new Mul;
  mul->type = type;
  mul->a = a;
  mul->b = b;
  return mul;
}

Expr Div::make(Expr a, Expr b) {
  return Div::make(a, b, max_type(a, b));
}

Expr Div::make(Expr a, Expr b, ComponentType type) {
  iassert(a.type() != typeOf<bool>()) << "Can't do arithmetic on booleans.";
  iassert(b.type() != typeOf<bool>()) << "Can't do arithmetic on booleans.";

  Div *div = new Div;
  div->type = type;
  div->a = a;
  div->b = b;
  return div;
}

Expr Rem::make(Expr a, Expr b) {
  return Rem::make(a, b, max_type(a, b));
}

Expr Rem::make(Expr a, Expr b, ComponentType type) {
  iassert(a.type() != typeOf<bool>()) << "Can't do arithmetic on booleans.";
  iassert(b.type() != typeOf<bool>()) << "Can't do arithmetic on booleans.";

  Rem *rem = new Rem;
  rem->type = type;
  rem->a = a;
  rem->b = b;
  return rem;
}

Expr Min::make(Expr a, Expr b) {
  return Min::make(a, b, max_type(a, b));
}

Expr Min::make(Expr a, Expr b, ComponentType type) {
  iassert(a.type() != typeOf<bool>()) << "Can't do arithmetic on booleans.";
  iassert(b.type() != typeOf<bool>()) << "Can't do arithmetic on booleans.";

  Min *min = new Min;
  min->type = type;
  min->a = a;
  min->b = b;
  return min;
}

Expr Max::make(Expr a, Expr b) {
  return Max::make(a, b, max_type(a, b));
}

Expr Max::make(Expr a, Expr b, ComponentType type) {
  iassert(a.type() != typeOf<bool>()) << "Can't do arithmetic on booleans.";
  iassert(b.type() != typeOf<bool>()) << "Can't do arithmetic on booleans.";

  Max *max = new Max;
  max->type = type;
  max->a = a;
  max->b = b;
  return max;
}

// Boolean binary ops
Expr Eq::make(Expr a, Expr b) {
  Eq *eq = new Eq;
  eq->type = typeOf<bool>();
  eq->a = a;
  eq->b = b;
  return eq;
}

Expr Neq::make(Expr a, Expr b) {
  Neq *neq = new Neq;
  neq->type = typeOf<bool>();
  neq->a = a;
  neq->b = b;
  return neq;
}

Expr Gt::make(Expr a, Expr b) {
  Gt *gt = new Gt;
  gt->type = typeOf<bool>();
  gt->a = a;
  gt->b = b;
  return gt;
}

Expr Lt::make(Expr a, Expr b) {
  Lt *lt = new Lt;
  lt->type = typeOf<bool>();
  lt->a = a;
  lt->b = b;
  return lt;
}

Expr Gte::make(Expr a, Expr b) {
  Gte *gte = new Gte;
  gte->type = typeOf<bool>();
  gte->a = a;
  gte->b = b;
  return gte;
}

Expr Lte::make(Expr a, Expr b) {
  Lte *lte = new Lte;
  lte->type = typeOf<bool>();
  lte->a = a;
  lte->b = b;
  return lte;
}

Expr Or::make(Expr a, Expr b) {
  Or *ornode = new Or;
  ornode->type = typeOf<bool>();
  ornode->a = a;
  ornode->b = b;
  return ornode;
}

Expr And::make(Expr a, Expr b) {
  And *andnode = new And;
  andnode->type = typeOf<bool>();
  andnode->a = a;
  andnode->b = b;
  return andnode;
}

// Load from an array
Expr Load::make(Expr arr) {
  return Load::make(arr, Literal::make(0));
}

Expr Load::make(Expr arr, Expr loc) {
  iassert(loc.type() == typeOf<int>()) << "Can't load from a non-integer offset";

  Load *load = new Load;
  load->type = arr.type();
  load->arr = arr;
  load->loc = loc;
  return load;
}

// Block
Stmt Block::make() {
  return Block::make({});
}

Stmt Block::make(std::vector<Stmt> b) {
  Block *block = new Block;
  block->contents = b;
  return block;
}

// Store to an array
Stmt Store::make(Expr arr, Expr loc, Expr data) {
  Store *store = new Store;
  store->arr = arr;
  store->loc = loc;
  store->data = data;
  return store;
}

// Conditional
Stmt IfThenElse::make(Expr cond, Stmt then) {
  return IfThenElse::make(cond, then, Block::make());
}

Stmt IfThenElse::make(Expr cond, Stmt then, Stmt otherwise) {
  iassert(cond.type() == typeOf<bool>()) << "Can only branch on boolean";
  
  IfThenElse* ite = new IfThenElse;
  ite->cond = cond;
  ite->then = then;
  ite->otherwise = otherwise;
  return ite;
}

// For loop
Stmt For::make(Expr var, Expr start, Expr end, Expr increment, Stmt contents,
  LoopKind kind, int vec_width) {
  For *loop = new For;
  loop->var = var;
  loop->start = start;
  loop->end = end;
  loop->increment = increment;
  loop->contents = contents;
  loop->kind = kind;
  loop->vec_width = vec_width;
  return loop;
}

// While loop
Stmt While::make(Expr cond, Stmt contents, LoopKind kind,
  int vec_width) {
  While *loop = new While;
  loop->cond = cond;
  loop->contents = contents;
  loop->kind = kind;
  loop->vec_width = vec_width;
  return loop;
}

// Function
Stmt Function::make(std::string name, std::vector<Expr> inputs,
  std::vector<Expr> outputs, Stmt body) {
  Function *func = new Function;
  func->name = name;
  func->body = body;
  func->inputs = inputs;
  func->outputs = outputs;
  return func;
}

// VarAssign
Stmt VarAssign::make(Expr lhs, Expr rhs) {
  iassert(lhs.as<Var>()) << "Can only assign to a Var";
  VarAssign *assign = new VarAssign;
  assign->lhs = lhs;
  assign->rhs = rhs;
  return assign;
}

// Allocate
Stmt Allocate::make(Expr var, Expr num_elements) {
  iassert(var.as<Var>() && var.as<Var>()->is_ptr) << "Can only allocate memory for a pointer-typed Var";
  iassert(num_elements.type() == typeOf<int>()) << "Can only allocate an integer-valued number of elements";
  Allocate* alloc = new Allocate;
  alloc->var = var;
  alloc->num_elements = num_elements;
  return alloc;
}

// Comment
Stmt Comment::make(std::string text) {
  Comment* comment = new Comment;
  comment->text = text;
  return comment;
}

// BlankLine
// TODO:make this a singleton
Stmt BlankLine::make() {
  return new BlankLine;
}

// Print
Stmt Print::make(std::string fmt, std::vector<Expr> params) {
  Print* pr = new Print;
  pr->fmt = fmt;
  pr->params = params;
  return pr;
}


// visitor methods
template<> void ExprNode<Literal>::accept(IRVisitor *v) const { v->visit((const Literal*)this); }
template<> void ExprNode<Var>::accept(IRVisitor *v) const { v->visit((const Var*)this); }
template<> void ExprNode<Add>::accept(IRVisitor *v) const { v->visit((const Add*)this); }
template<> void ExprNode<Sub>::accept(IRVisitor *v) const { v->visit((const Sub*)this); }
template<> void ExprNode<Mul>::accept(IRVisitor *v) const { v->visit((const Mul*)this); }
template<> void ExprNode<Div>::accept(IRVisitor *v) const { v->visit((const Div*)this); }
template<> void ExprNode<Rem>::accept(IRVisitor *v) const { v->visit((const Rem*)this); }
template<> void ExprNode<Min>::accept(IRVisitor *v) const { v->visit((const Min*)this); }
template<> void ExprNode<Max>::accept(IRVisitor *v) const { v->visit((const Max*)this); }
template<> void ExprNode<Eq>::accept(IRVisitor *v) const { v->visit((const Eq*)this); }
template<> void ExprNode<Neq>::accept(IRVisitor *v) const { v->visit((const Neq*)this); }
template<> void ExprNode<Gt>::accept(IRVisitor *v) const { v->visit((const Gt*)this); }
template<> void ExprNode<Lt>::accept(IRVisitor *v) const { v->visit((const Lt*)this); }
template<> void ExprNode<Gte>::accept(IRVisitor *v) const { v->visit((const Gte*)this); }
template<> void ExprNode<Lte>::accept(IRVisitor *v) const { v->visit((const Lte*)this); }
template<> void ExprNode<And>::accept(IRVisitor *v) const { v->visit((const And*)this); }
template<> void ExprNode<Or>::accept(IRVisitor *v) const { v->visit((const Or*)this); }
template<> void StmtNode<IfThenElse>::accept(IRVisitor *v) const { v->visit((const IfThenElse*)this); }
template<> void ExprNode<Load>::accept(IRVisitor *v) const { v->visit((const Load*)this); }
template<> void StmtNode<Store>::accept(IRVisitor *v) const { v->visit((const Store*)this); }
template<> void StmtNode<For>::accept(IRVisitor *v) const { v->visit((const For*)this); }
template<> void StmtNode<While>::accept(IRVisitor *v) const { v->visit((const While*)this); }
template<> void StmtNode<Block>::accept(IRVisitor *v) const { v->visit((const Block*)this); }
template<> void StmtNode<Function>::accept(IRVisitor *v) const { v->visit((const Function*)this); }
template<> void StmtNode<VarAssign>::accept(IRVisitor *v) const { v->visit((const VarAssign*)this); }
template<> void StmtNode<Allocate>::accept(IRVisitor *v) const { v->visit((const Allocate*)this); }
template<> void StmtNode<Comment>::accept(IRVisitor *v) const { v->visit((const Comment*)this); }
template<> void StmtNode<BlankLine>::accept(IRVisitor *v) const { v->visit((const BlankLine*)this); }
template<> void StmtNode<Print>::accept(IRVisitor *v) const { v->visit((const Print*)this); }

// printing methods
std::ostream &operator<<(std::ostream &os, const Stmt &op) {
  IRPrinter printer(os);
  op.accept(&printer);
  return os;
}

std::ostream &operator<<(std::ostream &os, const Expr &op) {
  IRPrinter printer(os);
  op.accept(&printer);
  return os;

}

} // namespace internal
} // namespace taco
