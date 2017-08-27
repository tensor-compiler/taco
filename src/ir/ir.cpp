#include "taco/ir/ir.h"
#include "taco/ir/ir_visitor.h"
#include "taco/ir/ir_printer.h"

#include "taco/error.h"
#include "taco/util/strings.h"

namespace taco {
namespace ir {

// class Expr
Expr::Expr(int n) : IRHandle(Literal::make(n)) {
}

Expr::Expr(float n) : IRHandle(Literal::make(n, Type(Type::Float, 32))) {
}

Expr::Expr(double n) : IRHandle(Literal::make(n, Type(Type::Float, 64))) {
}

Expr Literal::make(bool val) {
  Literal *lit = new Literal;
  lit->type = Type(Type::Bool);
  lit->value = val;
  return lit;
}

Expr Literal::make(int val) {
  Literal *lit = new Literal;
  lit->type = taco::type<int>();
  lit->value = (int64_t)val;
  return lit;
}

Expr Literal::make(double val, Type type) {
  Literal *lit = new Literal;
  lit->type = type;
  lit->dbl_value = val;
  return lit;
}

Expr Var::make(std::string name, Type type, bool is_ptr) {
  Var *var = new Var;
  var->type = type;
  var->name = name;

  // TODO: is_ptr and is_tensor should be part of type
  var->is_ptr = is_ptr;
  var->is_tensor = false;

  return var;
}

Expr Var::make(std::string name, Type type, Format format) {
  Var *var = new Var;
  var->name = name;
  var->type = type;
  var->format = format;
  var->is_ptr = var->is_tensor = true;
  return var;
}

Expr Neg::make(Expr a) {
  Neg *neg = new Neg;
  neg->a = a;
  neg->type = a.type();
  return neg;
}

Expr Sqrt::make(Expr a) {
  Sqrt *sqrt = new Sqrt;
  sqrt->a = a;
  sqrt->type = a.type();
  return sqrt;
}

// Binary Expressions
// helper
Type max_type(Expr a, Expr b);
Type max_type(Expr a, Expr b) {
  taco_iassert(!a.type().isBool() && !b.type().isBool()) <<
      "Can't do arithmetic on booleans.";

  if (a.type() == b.type()) {
    return a.type();
  } else {
    if (a.type() == Float(64) || b.type() == Float(64)) {
      return Float(64);
    }
    else {
      return Float(32);
    }
  }
}

Expr Add::make(Expr a, Expr b) {
  return Add::make(a, b, max_type(a, b));
}

Expr Add::make(Expr a, Expr b, Type type) {
  Add *add = new Add;
  add->type = type;
  add->a = a;
  add->b = b;
  return add;
}

Expr Sub::make(Expr a, Expr b) {
  return Sub::make(a, b, max_type(a, b));
}

Expr Sub::make(Expr a, Expr b, Type type) {
  taco_iassert(!a.type().isBool() && !b.type().isBool()) <<
      "Can't do arithmetic on booleans.";

  Sub *sub = new Sub;
  sub->type = type;
  sub->a = a;
  sub->b = b;
  return sub;
}

Expr Mul::make(Expr a, Expr b) {
  return Mul::make(a, b, max_type(a, b));
}

Expr Mul::make(Expr a, Expr b, Type type) {
  taco_iassert(!a.type().isBool() && !b.type().isBool()) <<
      "Can't do arithmetic on booleans.";

  Mul *mul = new Mul;
  mul->type = type;
  mul->a = a;
  mul->b = b;
  return mul;
}

Expr Div::make(Expr a, Expr b) {
  return Div::make(a, b, max_type(a, b));
}

Expr Div::make(Expr a, Expr b, Type type) {
  taco_iassert(!a.type().isBool() && !b.type().isBool()) <<
      "Can't do arithmetic on booleans.";

  Div *div = new Div;
  div->type = type;
  div->a = a;
  div->b = b;
  return div;
}

Expr Rem::make(Expr a, Expr b) {
  return Rem::make(a, b, max_type(a, b));
}

Expr Rem::make(Expr a, Expr b, Type type) {
  taco_iassert(!a.type().isBool() && !b.type().isBool()) <<
      "Can't do arithmetic on booleans.";

  Rem *rem = new Rem;
  rem->type = type;
  rem->a = a;
  rem->b = b;
  return rem;
}

Expr Min::make(Expr a, Expr b) {
  return Min::make({a, b}, max_type(a, b));
}

Expr Min::make(Expr a, Expr b, Type type) {
  return Min::make({a, b}, type);
}

Expr Min::make(std::vector<Expr> operands) {
  taco_iassert(operands.size() > 0);
  return Min::make(operands, operands[0].type());
}

Expr Min::make(std::vector<Expr> operands, Type type) {
  Min* min = new Min;
  min->operands = operands;
  min->type = type;
  return min;
}

Expr Max::make(Expr a, Expr b) {
  return Max::make(a, b, max_type(a, b));
}

Expr Max::make(Expr a, Expr b, Type type) {
  taco_iassert(!a.type().isBool() && !b.type().isBool()) <<
      "Can't do arithmetic on booleans.";

  Max *max = new Max;
  max->type = type;
  max->a = a;
  max->b = b;
  return max;
}

Expr BitAnd::make(Expr a, Expr b) {
  BitAnd *bitAnd = new BitAnd;
  bitAnd->type = Type(Type::UInt);
  bitAnd->a = a;
  bitAnd->b = b;
  return bitAnd;
}

// Boolean binary ops
Expr Eq::make(Expr a, Expr b) {
  Eq *eq = new Eq;
  eq->type = Bool();
  eq->a = a;
  eq->b = b;
  return eq;
}

Expr Neq::make(Expr a, Expr b) {
  Neq *neq = new Neq;
  neq->type = Bool();
  neq->a = a;
  neq->b = b;
  return neq;
}

Expr Gt::make(Expr a, Expr b) {
  Gt *gt = new Gt;
  gt->type = Bool();
  gt->a = a;
  gt->b = b;
  return gt;
}

Expr Lt::make(Expr a, Expr b) {
  Lt *lt = new Lt;
  lt->type = Bool();
  lt->a = a;
  lt->b = b;
  return lt;
}

Expr Gte::make(Expr a, Expr b) {
  Gte *gte = new Gte;
  gte->type = Bool();
  gte->a = a;
  gte->b = b;
  return gte;
}

Expr Lte::make(Expr a, Expr b) {
  Lte *lte = new Lte;
  lte->type = Bool();
  lte->a = a;
  lte->b = b;
  return lte;
}

Expr Or::make(Expr a, Expr b) {
  Or *ornode = new Or;
  ornode->type = Bool();
  ornode->a = a;
  ornode->b = b;
  return ornode;
}

Expr And::make(Expr a, Expr b) {
  And *andnode = new And;
  andnode->type = Bool();
  andnode->a = a;
  andnode->b = b;
  return andnode;
}

// Load from an array
Expr Load::make(Expr arr) {
  return Load::make(arr, Literal::make(0));
}

Expr Load::make(Expr arr, Expr loc) {
  taco_iassert(loc.type().isInt()) << "Can't load from a non-integer offset";
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

// Scope
Stmt Scope::make(Stmt scopedStmt) {
  Scope *scope = new Scope;
  scope->scopedStmt = scopedStmt;
  return scope;
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
  return IfThenElse::make(cond, then, Stmt());
}

Stmt IfThenElse::make(Expr cond, Stmt then, Stmt otherwise) {
  taco_iassert(then.defined());
  taco_iassert(cond.defined());
  taco_iassert(cond.type().isBool()) << "Can only branch on boolean";

  IfThenElse* ite = new IfThenElse;
  ite->cond = cond;
  ite->then = then;
  ite->otherwise = otherwise;
  ite->then = Scope::make(then);
  ite->otherwise = otherwise.defined() ? Scope::make(otherwise) : otherwise;
  return ite;
}

Stmt Case::make(std::vector<std::pair<Expr,Stmt>> clauses, bool alwaysMatch) {
  for (auto clause : clauses) {
    taco_iassert(clause.first.type().isBool()) << "Can only branch on boolean";
  }

  std::vector<std::pair<Expr,Stmt>> scopedClauses;
  for (auto& clause : clauses) {
    scopedClauses.push_back({clause.first, Scope::make(clause.second)});
  }
  
  Case* cs = new Case;
  cs->clauses = scopedClauses;
  cs->alwaysMatch = alwaysMatch;
  return cs;
}

// For loop
Stmt For::make(Expr var, Expr start, Expr end, Expr increment, Stmt contents,
  LoopKind kind, int vec_width) {
  For *loop = new For;
  loop->var = var;
  loop->start = start;
  loop->end = end;
  loop->increment = increment;
  loop->contents = Scope::make(contents);
  loop->kind = kind;
  loop->vec_width = vec_width;
  return loop;
}

// While loop
Stmt While::make(Expr cond, Stmt contents, LoopKind kind,
  int vec_width) {
  While *loop = new While;
  loop->cond = cond;
  loop->contents = Scope::make(contents);
  loop->kind = kind;
  loop->vec_width = vec_width;
  return loop;
}

// Function
Stmt Function::make(std::string name, std::vector<Expr> inputs,
  std::vector<Expr> outputs, Stmt body) {
  Function *func = new Function;
  func->name = name;
  func->body = Scope::make(body);
  func->inputs = inputs;
  func->outputs = outputs;
  return func;
}

// VarAssign
Stmt VarAssign::make(Expr lhs, Expr rhs, bool is_decl) {
  taco_iassert(lhs.as<Var>() || lhs.as<GetProperty>())
    << "Can only assign to a Var or GetProperty";
  VarAssign *assign = new VarAssign;
  assign->lhs = lhs;
  assign->rhs = rhs;
  assign->is_decl = is_decl;
  return assign;
}

// Allocate
Stmt Allocate::make(Expr var, Expr num_elements, bool is_realloc) {
  taco_iassert(var.as<GetProperty>() ||
               (var.as<Var>() && var.as<Var>()->is_ptr)) <<
      "Can only allocate memory for a pointer-typed Var";
  taco_iassert(num_elements.type().isInt()) <<
      "Can only allocate an integer-valued number of elements";
  Allocate* alloc = new Allocate;
  alloc->var = var;
  alloc->num_elements = num_elements;
  alloc->is_realloc = is_realloc;
  return alloc;
}

// Comment
Stmt Comment::make(std::string text) {
  Comment* comment = new Comment;
  comment->text = text;
  return comment;
}

// BlankLine
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
  
Expr GetProperty::make(Expr tensor, TensorProperty property, int dimension,
                       int index, std::string name) {
  GetProperty* gp = new GetProperty;
  gp->tensor = tensor;
  gp->property = property;
  gp->dimension = dimension;
  gp->name = name;
  gp->index = index;
  
  //TODO: deal with the fact that some of these are pointers
  if (property == TensorProperty::Values)
    gp->type = tensor.type();
  else
    gp->type = Type::Int;
  
  return gp;
}


// GetProperty
Expr GetProperty::make(Expr tensor, TensorProperty property, int dimension) {
  GetProperty* gp = new GetProperty;
  gp->tensor = tensor;
  gp->property = property;
  gp->dimension = dimension;
  
  //TODO: deal with the fact that these are pointers.
  if (property == TensorProperty::Values)
    gp->type = tensor.type();
  else
    gp->type = Type::Int;
  
  const Var* tensorVar = tensor.as<Var>();
  switch (property) {
    case TensorProperty::ComponentSize:
      gp->name = tensorVar->name + "_csize";
      break;
    case TensorProperty::DimensionOrder:
      gp->name = tensorVar->name  + util::toString(dimension + 1) + "_dim_order";
      break;
    case TensorProperty::Dimensions:
      gp->name = tensorVar->name + util::toString(dimension + 1) + "_size";
      break;
    case TensorProperty::Indices:
      taco_ierror << "Must provide both dimension and index for the Indices property";
      break;
    case TensorProperty::DimensionTypes:
      gp->name = tensorVar->name  + util::toString(dimension + 1) + "_dim_type";
      break;
    case TensorProperty::Order:
      gp->name = tensorVar->name + "_order";
      break;
    case TensorProperty::Values:
      gp->name = tensorVar->name + "_vals";
      break;
  }
  
  return gp;
}
  
// visitor methods
template<> void ExprNode<Literal>::accept(IRVisitorStrict *v)
    const { v->visit((const Literal*)this); }
template<> void ExprNode<Var>::accept(IRVisitorStrict *v)
    const { v->visit((const Var*)this); }
template<> void ExprNode<Neg>::accept(IRVisitorStrict *v)
    const { v->visit((const Neg*)this); }
template<> void ExprNode<Sqrt>::accept(IRVisitorStrict *v)
    const { v->visit((const Sqrt*)this); }
template<> void ExprNode<Add>::accept(IRVisitorStrict *v)
    const { v->visit((const Add*)this); }
template<> void ExprNode<Sub>::accept(IRVisitorStrict *v)
    const { v->visit((const Sub*)this); }
template<> void ExprNode<Mul>::accept(IRVisitorStrict *v)
    const { v->visit((const Mul*)this); }
template<> void ExprNode<Div>::accept(IRVisitorStrict *v)
    const { v->visit((const Div*)this); }
template<> void ExprNode<Rem>::accept(IRVisitorStrict *v)
    const { v->visit((const Rem*)this); }
template<> void ExprNode<Min>::accept(IRVisitorStrict *v)
    const { v->visit((const Min*)this); }
template<> void ExprNode<Max>::accept(IRVisitorStrict *v)
    const { v->visit((const Max*)this); }
template<> void ExprNode<BitAnd>::accept(IRVisitorStrict *v)
    const { v->visit((const BitAnd*)this); }
template<> void ExprNode<Eq>::accept(IRVisitorStrict *v)
    const { v->visit((const Eq*)this); }
template<> void ExprNode<Neq>::accept(IRVisitorStrict *v)
    const { v->visit((const Neq*)this); }
template<> void ExprNode<Gt>::accept(IRVisitorStrict *v)
    const { v->visit((const Gt*)this); }
template<> void ExprNode<Lt>::accept(IRVisitorStrict *v)
    const { v->visit((const Lt*)this); }
template<> void ExprNode<Gte>::accept(IRVisitorStrict *v)
    const { v->visit((const Gte*)this); }
template<> void ExprNode<Lte>::accept(IRVisitorStrict *v)
    const { v->visit((const Lte*)this); }
template<> void ExprNode<And>::accept(IRVisitorStrict *v)
    const { v->visit((const And*)this); }
template<> void ExprNode<Or>::accept(IRVisitorStrict *v)
    const { v->visit((const Or*)this); }
template<> void StmtNode<IfThenElse>::accept(IRVisitorStrict *v)
    const { v->visit((const IfThenElse*)this); }
template<> void StmtNode<Case>::accept(IRVisitorStrict *v)
    const { v->visit((const Case*)this); }
template<> void ExprNode<Load>::accept(IRVisitorStrict *v)
    const { v->visit((const Load*)this); }
template<> void StmtNode<Store>::accept(IRVisitorStrict *v)
    const { v->visit((const Store*)this); }
template<> void StmtNode<For>::accept(IRVisitorStrict *v)
    const { v->visit((const For*)this); }
template<> void StmtNode<While>::accept(IRVisitorStrict *v)
    const { v->visit((const While*)this); }
template<> void StmtNode<Block>::accept(IRVisitorStrict *v)
    const { v->visit((const Block*)this); }
template<> void StmtNode<Scope>::accept(IRVisitorStrict *v)
    const { v->visit((const Scope*)this); }
template<> void StmtNode<Function>::accept(IRVisitorStrict *v)
    const { v->visit((const Function*)this); }
template<> void StmtNode<VarAssign>::accept(IRVisitorStrict *v)
    const { v->visit((const VarAssign*)this); }
template<> void StmtNode<Allocate>::accept(IRVisitorStrict *v)
    const { v->visit((const Allocate*)this); }
template<> void StmtNode<Comment>::accept(IRVisitorStrict *v)
    const { v->visit((const Comment*)this); }
template<> void StmtNode<BlankLine>::accept(IRVisitorStrict *v)
    const { v->visit((const BlankLine*)this); }
template<> void StmtNode<Print>::accept(IRVisitorStrict *v)
    const { v->visit((const Print*)this); }
template<> void ExprNode<GetProperty>::accept(IRVisitorStrict *v)
    const { v->visit((const GetProperty*)this); }

// printing methods
std::ostream& operator<<(std::ostream& os, const Stmt& stmt) {
  if (!stmt.defined()) return os << "Stmt()";
  IRPrinter printer(os);
  stmt.accept(&printer);
  return os;
}

std::ostream& operator<<(std::ostream& os, const Expr& expr) {
  if (!expr.defined()) return os << "Expr()";
  IRPrinter printer(os);
  expr.accept(&printer);
  return os;
}

} // namespace ir
} // namespace taco
