#include "taco/ir/ir.h"
#include "taco/ir/ir_visitor.h"
#include "taco/ir/ir_rewriter.h"
#include "taco/ir/ir_printer.h"
#include "taco/index_notation/index_notation.h"

#include "taco/error.h"
#include "taco/util/strings.h"
#include "taco/type.h"
#include "taco/spatial.h"
#include <execinfo.h>

namespace taco {
namespace ir {

// class Expr
Expr::Expr(bool n) : IRHandle(Literal::make(n)) {
}

Expr::Expr(int8_t n) : IRHandle(Literal::make(n)) {
}

Expr::Expr(int16_t n) : IRHandle(Literal::make(n)) {
}

Expr::Expr(int32_t n) : IRHandle(Literal::make(n)) {
}

Expr::Expr(int64_t n) : IRHandle(Literal::make(n)) {
}

Expr::Expr(uint8_t n) : IRHandle(Literal::make(n)) {
}

Expr::Expr(uint16_t n) : IRHandle(Literal::make(n)) {
}

Expr::Expr(uint32_t n) : IRHandle(Literal::make(n)) {
}

Expr::Expr(uint64_t n) : IRHandle(Literal::make(n)) {
}

Expr::Expr(float n) : IRHandle(Literal::make(n)) {
}

Expr::Expr(double n) : IRHandle(Literal::make(n)) {
}

Expr::Expr(std::complex<float> n) : IRHandle(Literal::make(n)) {
}

Expr::Expr(std::complex<double> n) : IRHandle(Literal::make(n)) {
}

Expr Literal::zero(Datatype datatype) {
  Expr zero;
  switch (datatype.getKind()) {
    case Datatype::Bool:
      zero = Literal::make(false);
      break;
    case Datatype::UInt8:
      zero = Literal::make((uint8_t)0);
      break;
    case Datatype::UInt16:
      zero = Literal::make((uint16_t)0);
      break;
    case Datatype::UInt32:
      zero = Literal::make((uint32_t)0);
      break;
    case Datatype::UInt64:
      zero = Literal::make((uint64_t)0);
      break;
    case Datatype::UInt128:
      taco_not_supported_yet;
      break;
    case Datatype::Int8:
      zero = Literal::make((int8_t)0);
      break;
    case Datatype::Int16:
      zero = Literal::make((int16_t)0);
      break;
    case Datatype::Int32:
      zero = Literal::make((int32_t)0);
      break;
    case Datatype::Int64:
      zero = Literal::make((int64_t)0);
      break;
    case Datatype::Int128:
      taco_not_supported_yet;
      break;
    case Datatype::Float32:
      zero = Literal::make((float)0.0);
      break;
    case Datatype::Float64:
      zero = Literal::make((double)0.0);
      break;
    case Datatype::Complex64:
      zero = Literal::make(std::complex<float>());
      break;
    case Datatype::Complex128:
      zero = Literal::make(std::complex<double>());
      break;
    case Datatype::Undefined:
      taco_ierror;
      break;
  }
    taco_iassert(zero.defined());
    return zero;
}

Literal::~Literal() {
  free(value.get());
}

bool Literal::getBoolValue() const {
  taco_iassert(type.isBool()) << "Type must be boolean";
  return getValue<bool>();
}

int64_t Literal::getIntValue() const {
  taco_iassert(type.isInt()) << "Type must be integer";
  switch (type.getKind()) {
    case Datatype::Int8:
      return getValue<int8_t>();
    case Datatype::Int16:
      return getValue<int16_t>();
    case Datatype::Int32:
      return getValue<int32_t>();
    case Datatype::Int64:
      return getValue<int64_t>();
    case Datatype::Int128:
      taco_not_supported_yet;
    default:
      break;
  }
  taco_ierror << "not an integer type";
  return 0ll;
}

uint64_t Literal::getUIntValue() const {
  taco_iassert(type.isUInt()) << "Type must be unsigned integer";
  switch (type.getKind()) {
    case Datatype::UInt8:
      return getValue<uint8_t>();
    case Datatype::UInt16:
      return getValue<uint16_t>();
    case Datatype::UInt32:
      return getValue<uint32_t>();
    case Datatype::UInt64:
      return getValue<uint64_t>();
    case Datatype::UInt128:
      taco_not_supported_yet;
    default:
      break;
  }
  taco_ierror << "not an unsigned integer type";
  return 0ull;
}

double Literal::getFloatValue() const {
  taco_iassert(type.isFloat()) << "Type must be floating point";
  switch (type.getKind()) {
    case Datatype::Float32:
      static_assert(sizeof(float) == 4, "Float not 32 bits");
      return getValue<float>();
    case Datatype::Float64:
      return getValue<double>();
    default:
      break;
  }
  taco_ierror << "not a floating point type";
  return 0.0;
}

std::complex<double> Literal::getComplexValue() const {
  taco_iassert(type.isComplex()) << "Type must be a complex number";
  switch (type.getKind()) {
    case Datatype::Complex64:
      return getValue<std::complex<float>>();
    case Datatype::Complex128:
      return getValue<std::complex<double>>();
    default:
      break;
  }
  taco_ierror << "not a floating point type";
  return 0.0;
}

template <typename T> bool compare(const Literal* literal, double val) {
      return literal->getValue<T>() == static_cast<T>(val);
}

bool Literal::equalsScalar(double scalar) const {
  switch (type.getKind()) {
    case Datatype::Bool:
      return compare<bool>(this, scalar);
    break;
    case Datatype::UInt8:
      return compare<uint8_t>(this, scalar);
    break;
    case Datatype::UInt16:
      return compare<uint16_t>(this, scalar);
    break;
    case Datatype::UInt32:
      return compare<uint32_t>(this, scalar);
    break;
    case Datatype::UInt64:
      return compare<uint64_t>(this, scalar);
    break;
    case Datatype::UInt128:
      taco_not_supported_yet;
    break;
    case Datatype::Int8:
      return compare<int8_t>(this, scalar);
    break;
    case Datatype::Int16:
      return compare<int16_t>(this, scalar);
    break;
    case Datatype::Int32:
      return compare<int32_t>(this, scalar);
    break;
    case Datatype::Int64:
      return compare<int64_t>(this, scalar);
    break;
    case Datatype::Int128:
      taco_not_supported_yet;
    break;
    case Datatype::Float32:
      return compare<float>(this, scalar);
    break;
    case Datatype::Float64:
      return compare<double>(this, scalar);
    break;
    case Datatype::Complex64:
      return compare<std::complex<float>>(this, scalar);
    break;
    case Datatype::Complex128:
      return compare<std::complex<double>>(this, scalar);
    break;
    case Datatype::Undefined:
      taco_not_supported_yet;
    break;
  }
  return false;
}

Expr Var::make(std::string name, Datatype type, 
               bool is_ptr, bool is_tensor,
               bool is_parameter, MemoryLocation memoryLocation) {
  Var *var = new Var;
  var->type = type;
  var->name = name;

  // TODO: is_ptr and is_tensor should be part of type
  var->is_ptr = is_ptr;
  var->is_tensor = is_tensor;
  var->is_parameter = is_parameter;

  var->memoryLocation = memoryLocation;

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
Datatype max_expr_type(Expr a, Expr b);
Datatype max_expr_type(Expr a, Expr b) {
  return max_type(a.type(), b.type());
}

Expr Add::make(Expr a, Expr b) {
  return Add::make(a, b, max_expr_type(a, b));
}

Expr Add::make(Expr a, Expr b, Datatype type) {
  //if (a.type().isBool() || b.type().isBool()) {
  //  std::cout << a << std::endl;
  //  std::cout << b << std::endl;
  //  void* callstack[128];
  //  int i, frames = backtrace(callstack, 128);
  //  char** strs = backtrace_symbols(callstack, frames);
  //  for (i = 0; i < frames; ++i) {
  //    printf("%s\n", strs[i]);
  //  }
  //  free(strs);
  //}
  taco_iassert((!a.type().isBool() || (isa<Var>(a) && a.as<Var>()->is_ptr)) &&
            (!b.type().isBool() || (isa<Var>(b) && b.as<Var>()->is_ptr))) <<
      "Can't do arithmetic on booleans.";

  Add *add = new Add;
  add->type = type;
  add->a = a;
  add->b = b;
  return add;
}

Expr Sub::make(Expr a, Expr b) {
  return Sub::make(a, b, max_expr_type(a, b));
}

Expr Sub::make(Expr a, Expr b, Datatype type) {
  taco_iassert(!a.type().isBool() && !b.type().isBool()) <<
      "Can't do arithmetic on booleans.";
  
  Sub *sub = new Sub;
  sub->type = type;
  sub->a = a;
  sub->b = b;
  return sub;
}

Expr Mul::make(Expr a, Expr b) {
  return Mul::make(a, b, max_expr_type(a, b));
}

Expr Mul::make(Expr a, Expr b, Datatype type) {
  taco_iassert(!a.type().isBool() && !b.type().isBool()) <<
      "Can't do arithmetic on booleans.";
  
  Mul *mul = new Mul;
  mul->type = type;
  mul->a = a;
  mul->b = b;
  return mul;
}

Expr Div::make(Expr a, Expr b) {
  return Div::make(a, b, max_expr_type(a, b));
}

Expr Div::make(Expr a, Expr b, Datatype type) {
  taco_iassert(!a.type().isBool() && !b.type().isBool()) <<
      "Can't do arithmetic on booleans.";
  
  Div *div = new Div;
  div->type = type;
  div->a = a;
  div->b = b;
  return div;
}

Expr Rem::make(Expr a, Expr b) {
  return Rem::make(a, b, max_expr_type(a, b));
}

Expr Rem::make(Expr a, Expr b, Datatype type) {
  taco_iassert(!a.type().isBool() && !b.type().isBool()) <<
      "Can't do arithmetic on booleans.";
  
  Rem *rem = new Rem;
  rem->type = type;
  rem->a = a;
  rem->b = b;
  return rem;
}

Expr Min::make(Expr a, Expr b) {
  return Min::make({a, b}, max_expr_type(a, b));
}

Expr Min::make(Expr a, Expr b, Datatype type) {
  return Min::make({a, b}, type);
}

Expr Min::make(std::vector<Expr> operands) {
  taco_iassert(operands.size() > 0);
  return Min::make(operands, operands[0].type());
}

Expr Min::make(std::vector<Expr> operands, Datatype type) {
  Min* min = new Min;
  min->operands = operands;
  min->type = type;
  return min;
}

Expr Max::make(Expr a, Expr b) {
  return Max::make(a, b, max_expr_type(a, b));
}

Expr Max::make(Expr a, Expr b, Datatype type) {
  taco_iassert(!a.type().isBool() && !b.type().isBool()) <<
      "Can't do arithmetic on booleans.";

  return Max::make({a, b}, type);
}

Expr Max::make(std::vector<Expr> operands) {
  taco_iassert(operands.size() > 0);
  return Max::make(operands, operands[0].type());
}

Expr Max::make(std::vector<Expr> operands, Datatype type) {
  Max* max = new Max;
  max->operands = operands;
  max->type = type;
  return max;
}

Expr BitAnd::make(Expr a, Expr b) {
  BitAnd *bitAnd = new BitAnd;
  bitAnd->type = UInt();
  bitAnd->a = a;
  bitAnd->b = b;
  return bitAnd;
}

Expr BitOr::make(Expr a, Expr b) {
  BitOr *bitOr = new BitOr;
  bitOr->type = UInt();
  bitOr->a = a;
  bitOr->b = b;
  return bitOr;
}

// Boolean binary ops
Expr Eq::make(Expr a, Expr b) {
  Eq *eq = new Eq;
  eq->type = Bool;
  eq->a = a;
  eq->b = b;
  return eq;
}

Expr Neq::make(Expr a, Expr b) {
  Neq *neq = new Neq;
  neq->type = Bool;
  neq->a = a;
  neq->b = b;
  return neq;
}

Expr Gt::make(Expr a, Expr b) {
  Gt *gt = new Gt;
  gt->type = Bool;
  gt->a = a;
  gt->b = b;
  return gt;
}

Expr Lt::make(Expr a, Expr b) {
  Lt *lt = new Lt;
  lt->type = Bool;
  lt->a = a;
  lt->b = b;
  return lt;
}

Expr Gte::make(Expr a, Expr b) {
  Gte *gte = new Gte;
  gte->type = Bool;
  gte->a = a;
  gte->b = b;
  return gte;
}

Expr Lte::make(Expr a, Expr b) {
  Lte *lte = new Lte;
  lte->type = Bool;
  lte->a = a;
  lte->b = b;
  return lte;
}

Expr Or::make(Expr a, Expr b) {
  Or *ornode = new Or;
  ornode->type = Bool;
  ornode->a = a;
  ornode->b = b;
  return ornode;
}

Expr And::make(Expr a, Expr b) {
  And *andnode = new And;
  andnode->type = Bool;
  andnode->a = a;
  andnode->b = b;
  return andnode;
}

Expr Cast::make(Expr a, Datatype newType) {
  Cast *cast = new Cast;
  cast->type = newType;
  cast->a = a;
  return cast;
}

Expr Call::make(const std::string& func, const std::vector<Expr>& args, 
                Datatype type) {
  Call *call = new Call;
  call->type = type;
  call->func = func;
  call->args = args;
  return call;
}

// Load
Expr Load::make(Expr arr) {
  return Load::make(arr, Literal::make((int64_t)0));
}

Expr Load::make(Expr arr, Expr loc, MemoryLocation mem_loc) {
  taco_iassert(loc.type().isInt() || loc.type().isUInt()) 
      << "Can't load from a non-integer offset";
  Load *load = new Load;
  load->type = arr.type();
  load->arr = arr;
  load->loc = loc;
  load->mem_loc = mem_loc;
  return load;
}

// Malloc
Expr Malloc::make(Expr size, MemoryLocation memoryLocation) {
  taco_iassert(size.defined());
  Malloc *malloc = new Malloc;
  malloc->size = size;
  malloc->memoryLocation = memoryLocation;
  return malloc;
}

// Sizeof
Expr Sizeof::make(Type type) {
  Sizeof *szeof = new Sizeof;
  szeof->type = UInt64;
  szeof->sizeofType = type;
  return szeof;
}

// Block
Stmt Block::make() {
  return Block::make({});
}

static bool nop(const Stmt& stmt) {
  if (!stmt.defined()) return true;
  if (isa<Block>(stmt) && to<Block>(stmt)->contents.size() == 0) return true;
  return false;
}

Stmt Block::make(std::vector<Stmt> stmts) {
  Block *block = new Block;
  for (auto& stmt : stmts) {
    if (nop(stmt)) continue;
    block->contents.push_back(stmt);
  }
  return block;
}

Stmt Block::blanks(std::vector<Stmt> stmts) {
  Block *block = new Block;

  // Add first defined statement to result.
  size_t i = 0;
  for (; i < stmts.size(); i++) {
    Stmt stmt = stmts[i];
    if (nop(stmt)) continue;
    block->contents.push_back(stmt);
    break;
  }
  i++;

  // Add additional defined statements to result prefixed with a blank line.
  for (; i < stmts.size(); i++) {
    Stmt stmt = stmts[i];
    if (nop(stmt)) continue;
    block->contents.push_back(BlankLine::make());
    block->contents.push_back(stmt);
  }

  return block;
}

// Scope
Stmt Scope::make(Stmt scopedStmt) {
  taco_iassert(scopedStmt.defined());

  if (isa<Scope>(scopedStmt)) {
    return scopedStmt;
  }

  Scope *scope = new Scope;
  scope->scopedStmt = scopedStmt;
  return scope;
}

// Spatial Only
Stmt Scope::make(Stmt scopedStmt, Expr returnExpr) {
  taco_iassert(scopedStmt.defined());

  if (isa<Scope>(scopedStmt)) {
    return scopedStmt;
  }

  Scope *scope = new Scope;
  scope->scopedStmt = scopedStmt;
  scope->returnExpr = returnExpr;
  return scope;
}

// Store to an array
Stmt Store::make(Expr arr, Expr loc, Expr data, bool use_atomics, ParallelUnit atomic_parallel_unit) {
  Store *store = new Store;
  store->arr = arr;
  store->loc = loc;
  store->data = data;
  store->use_atomics = use_atomics;
  store->atomic_parallel_unit = atomic_parallel_unit;
  return store;
}

Stmt Store::make(Expr arr, Expr loc, Expr data, MemoryLocation lhsMemLoc, MemoryLocation rhsMemLoc, bool use_atomics, ParallelUnit atomic_parallel_unit) {
  Store *store = new Store;
  store->arr = arr;
  store->loc = loc;
  store->data = data;
  store->use_atomics = use_atomics;
  store->atomic_parallel_unit = atomic_parallel_unit;
  store->lhs_mem_loc = lhsMemLoc;
  store->rhs_mem_loc = rhsMemLoc;
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

Expr Ternary::make(Expr cond, Expr then, Expr otherwise) {
  taco_iassert(then.defined());
  taco_iassert(cond.defined());
  taco_iassert(cond.type().isBool()) << "Can only branch on boolean";

  Ternary* ternary = new Ternary;
  ternary->cond = cond;
  ternary->then = then;
  ternary->otherwise = otherwise;
  return ternary;
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

Stmt Switch::make(std::vector<std::pair<Expr,Stmt>> cases, Expr controlExpr) {
  for (auto switchCase : cases) {
    taco_iassert(switchCase.first.type().isUInt()) << "Can only switch on uint";
  }

  std::vector<std::pair<Expr,Stmt>> scopedCases;
  for (auto& switchCase : cases) {
    scopedCases.push_back({switchCase.first, Scope::make(switchCase.second)});
  }
  
  Switch* sw = new Switch;
  sw->cases = scopedCases;
  sw->controlExpr = controlExpr;
  return sw;
}

// For loop
Stmt For::make(Expr var, Expr start, Expr end, Expr increment, Stmt body,
  LoopKind kind, ParallelUnit parallel_unit, size_t unrollFactor, int vec_width, size_t numChunks) {
  For *loop = new For;
  loop->var = var;
  loop->start = start;
  loop->end = end;
  loop->increment = increment;
  loop->contents = Scope::make(body);
  loop->kind = kind;
  loop->unrollFactor = unrollFactor;
  loop->vec_width = vec_width;
  loop->parallel_unit = parallel_unit;
  loop->numChunks = numChunks;
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
Stmt Function::make(std::string name,
                    std::vector<Expr> outputs, std::vector<Expr> inputs,
                    Stmt body) {
  Function *func = new Function;
  func->name = name;
  func->body = Scope::make(body);
  func->inputs = inputs;
  func->outputs = outputs;
  return func;
}

std::pair<std::vector<Datatype>,Datatype> Function::getReturnType() const {
  struct InferReturnType : IRVisitor {
    std::pair<std::vector<Datatype>,Datatype> returnType;

    using IRVisitor::visit;

    void visit(const Yield* stmt) {
      if (returnType.second != Datatype()) {
        taco_iassert(returnType.second == stmt->val.type());
        taco_iassert(returnType.first.size() == stmt->coords.size());
        taco_iassert([&]() {
            for (size_t i = 0; i < stmt->coords.size(); ++i) {
              if (returnType.first[i] != stmt->coords[i].type()) {
                return false;
              }
            }
            return true;
          }()); 
        return;
      }
      for (auto& coord : stmt->coords) {
        returnType.first.push_back(coord.type());
      }
      returnType.second = stmt->val.type();
    }

    std::pair<std::vector<Datatype>,Datatype> inferType(Stmt stmt) {
      returnType = {{}, Datatype()};
      stmt.accept(this);
      return returnType;
    }
  };
  return InferReturnType().inferType(this);
}

// VarDecl
Stmt VarDecl::make(Expr var, Expr rhs, MemoryLocation mem) {
  taco_iassert(var.as<Var>())
    << "Can only declare a Var" << var << "=" << rhs << MemoryLocation_NAMES[(int)mem];
  VarDecl* decl = new VarDecl;
  decl->var = var;
  decl->rhs = rhs;
  decl->mem = mem;
  return decl;
}

// VarAssign
Stmt Assign::make(Expr lhs, Expr rhs, bool use_atomics, ParallelUnit atomic_parallel_unit) {
  taco_iassert(lhs.as<Var>() || lhs.as<GetProperty>())
    << "Can only assign to a Var or GetProperty";
  Assign *assign = new Assign;
  assign->lhs = lhs;
  assign->rhs = rhs;
  assign->use_atomics = use_atomics;
  assign->atomic_parallel_unit = atomic_parallel_unit;
  return assign;
}

// Yield
Stmt Yield::make(std::vector<Expr> coords, Expr val) {
  for (auto coord : coords) {
    taco_iassert(coord.as<Var>()) << "Coordinates must be instances of Var";
  }
  Yield *yield = new Yield;
  yield->coords = coords;
  yield->val = val;
  return yield;
}

// Allocate
Stmt Allocate::make(Expr var, Expr num_elements, bool is_realloc, Expr
                    old_elements, bool clear, MemoryLocation memoryLocation) {
  taco_iassert(var.as<GetProperty>() ||
               (var.as<Var>() && var.as<Var>()->is_ptr)) <<
      "Can only allocate memory for a pointer-typed Var";
  taco_iassert(num_elements.type().isInt() || num_elements.type().isUInt()) <<
      "Can only allocate an integer-valued number of elements";
  Allocate* alloc = new Allocate;
  alloc->var = var;
  alloc->num_elements = num_elements;
  alloc->is_realloc = is_realloc;
  taco_iassert(!is_realloc || old_elements.ptr != NULL);
  alloc->old_elements = old_elements;
  alloc->clear = clear;
  alloc->memoryLocation = memoryLocation;
  return alloc;
}

// Free
Stmt Free::make(Expr var) {
  taco_iassert(var.as<GetProperty>() ||
               (var.as<Var>() && var.as<Var>()->is_ptr)) <<
      "Can only allocate memory for a pointer-typed Var";
  Free* free = new Free;
  free->var = var;
  return free;
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

// Continue
Stmt Continue::make() {
  return new Continue;
}

// Break
Stmt Break::make() {
  return new Break;
}

// Print
Stmt Print::make(std::string fmt, std::vector<Expr> params) {
  Print* pr = new Print;
  pr->fmt = fmt;
  pr->params = params;
  return pr;
}
  
Expr GetProperty::make(Expr tensor, TensorProperty property, int mode,
                       int index, std::string name) {
  GetProperty* gp = new GetProperty;
  gp->tensor = tensor;
  gp->property = property;
  gp->mode = mode;
  gp->name = name;
  gp->index = index;
  
  //TODO: deal with the fact that some of these are pointers
  if (property == TensorProperty::Values)
    gp->type = tensor.type();
  else
    gp->type = Int();
  
  return gp;
}

// Sort
Stmt Sort::make(std::vector<Expr> args) {
  Sort* sort = new Sort;
  sort->args = args;
  return sort;
}

Expr GetProperty::make(Expr tensor, TensorProperty property, int mode, int index, bool is_compressed) {
  GetProperty* gp = new GetProperty;
  gp->tensor = tensor;
  gp->property = property;
  gp->mode = mode;
  gp->is_compressed = is_compressed;

  //TODO: deal with the fact that these are pointers.
  if (property == TensorProperty::Values)
    gp->type = tensor.type();
  else
    gp->type = Int();

  const Var* tensorVar = tensor.as<Var>();
  switch (property) {
    case TensorProperty::Order:
      gp->name = tensorVar->name + "_order";
      break;
    case TensorProperty::Dimension:
      gp->name = tensorVar->name + util::toString(mode + 1) + "_dimension";

      // Assign dimension value for spatial code
      if (should_use_Spatial_codegen())
        gp->index = index;

      break;
    case TensorProperty::ComponentSize:
      gp->name = tensorVar->name + "_csize";
      break;
    case TensorProperty::ModeOrdering:
      gp->name = tensorVar->name  + util::toString(mode + 1) + "_mode_ordering";
      break;
    case TensorProperty::ModeTypes:
      gp->name = tensorVar->name  + util::toString(mode + 1) + "_mode_type";
      break;
    case TensorProperty::Indices:
      taco_ierror << "Must provide both mode and index for the Indices property";
      break;
    case TensorProperty::Values:
      gp->name = tensorVar->name + "_vals";
      // Assign dimension value for spatial code
      if (should_use_Spatial_codegen())

        gp->index = index;
      break;
    case TensorProperty::ValuesSize:
      gp->name = tensorVar->name + "_vals_size";
      break;
  }

  return gp;
}

// GetProperty
Expr GetProperty::make(Expr tensor, TensorProperty property, int mode) {
  GetProperty* gp = new GetProperty;
  gp->tensor = tensor;
  gp->property = property;
  gp->mode = mode;

  //TODO: deal with the fact that these are pointers.
  if (property == TensorProperty::Values)
    gp->type = tensor.type();
  else
    gp->type = Int();
  
  const Var* tensorVar = tensor.as<Var>();
  switch (property) {
    case TensorProperty::Order:
      gp->name = tensorVar->name + "_order";
      break;
    case TensorProperty::Dimension:
      gp->name = tensorVar->name + util::toString(mode + 1) + "_dimension";

      // Assign dimension value for spatial code
      if (should_use_Spatial_codegen())
        gp->index = get_Spatial_dimension();
 
      break;
    case TensorProperty::ComponentSize:
      gp->name = tensorVar->name + "_csize";
      break;
    case TensorProperty::ModeOrdering:
      gp->name = tensorVar->name  + util::toString(mode + 1) + "_mode_ordering";
      break;
    case TensorProperty::ModeTypes:
      gp->name = tensorVar->name  + util::toString(mode + 1) + "_mode_type";
      break;
    case TensorProperty::Indices:
      taco_ierror << "Must provide both mode and index for the Indices property";
      break;
    case TensorProperty::Values:
      gp->name = tensorVar->name + "_vals";
      break;
    case TensorProperty::ValuesSize:
      gp->name = tensorVar->name + "_vals_size";
      break;
  }
  
  return gp;
}

// Bulk Store
Stmt StoreBulk::make(Expr arr, Expr locStart, Expr locEnd, Expr data, bool use_atomics, ParallelUnit atomic_parallel_unit) {
  StoreBulk *store = new StoreBulk;
  store->arr = arr;
  store->locStart = locStart;
  store->locEnd = locEnd;
  store->data = data;
  store->use_atomics = use_atomics;
  store->atomic_parallel_unit = atomic_parallel_unit;
  store->lhs_mem_loc = MemoryLocation::Default;
  store->rhs_mem_loc = MemoryLocation::Default;
  return store;
}

Stmt StoreBulk::make(Expr arr, Expr locStart, Expr locEnd, Expr data, MemoryLocation lhs_mem_loc, MemoryLocation rhs_mem_loc,
                     bool use_atomics, ParallelUnit atomic_parallel_unit) {
  StoreBulk *store = new StoreBulk;
  store->arr = arr;
  store->locStart = locStart;
  store->locEnd = locEnd;
  store->data = data;
  store->use_atomics = use_atomics;
  store->atomic_parallel_unit = atomic_parallel_unit;
  store->lhs_mem_loc = lhs_mem_loc;
  store->rhs_mem_loc = rhs_mem_loc;
  return store;
}

// Bulk Load
Expr LoadBulk::make(Expr arr, Expr locStart, Expr locEnd) {
  taco_iassert(locStart.type().isInt() || locStart.type().isUInt())
          << "Can't load from a non-integer offset";
  taco_iassert(locEnd.type().isInt() || locEnd.type().isUInt())
          << "Can't load from a non-integer offset";
  LoadBulk *load = new LoadBulk;
  load->type = arr.type();
  load->arr = arr;
  load->locStart = locStart;
  load->locEnd = locEnd;
  return load;
}

/// SPATIAL ONLY
// Reduce loop
Stmt Reduce::make(Expr var, Expr reg, Expr start, Expr end, Expr increment, Stmt body,
                bool add, size_t par) {
  Reduce *loop = new Reduce;
  loop->var = var;
  loop->reg = reg;
  loop->start = start;
  loop->end = end;
  loop->increment = increment;
  loop->contents = body;
  loop->add = add;
  loop->par = par;
  return loop;
}

Stmt Reduce::make(Expr var, Expr reg, Expr start, Expr end, Expr increment, Stmt body, Expr returnExpr,
                  bool add, size_t par) {
  Reduce *loop = new Reduce;
  loop->var = var;
  loop->reg = reg;
  loop->start = start;
  loop->end = end;
  loop->increment = increment;
  loop->contents = Scope::make(body);
  loop->returnExpr = returnExpr;
  loop->add = add;
  loop->par = par;
  return loop;
}

// Reduce loop with a scanner as the iteration pattern
Stmt ReduceScan::make(Expr caseType, Expr reg, Expr scanner, Stmt body,
                  bool add) {
  ReduceScan *loop = new ReduceScan;
  loop->caseType = caseType;
  loop->reg = reg;
  loop->scanner = scanner;
  loop->contents = body;
  loop->add = add;
  return loop;
}

Stmt ReduceScan::make(Expr caseType, Expr reg, Expr scanner, Stmt contents, Expr returnExpr,
                  bool add) {
  ReduceScan *loop = new ReduceScan;
  loop->caseType = caseType;
  loop->reg = reg;
  loop->scanner = scanner;
  if (contents.defined())
    loop->contents = Scope::make(contents);
  loop->returnExpr = returnExpr;
  loop->add = add;
  return loop;
}

// For loop with a scanner as the iteration pattern
Stmt ForScan::make(Expr caseType, Expr scanner, Stmt body,
               LoopKind kind, ParallelUnit parallel_unit, size_t unrollFactor, int vec_width, size_t numChunks) {
  ForScan *loop = new ForScan;
  loop->caseType = caseType;
  loop->scanner = scanner;
  loop->contents = Scope::make(body);
  loop->kind = kind;
  loop->unrollFactor = unrollFactor;
  loop->vec_width = vec_width;
  loop->parallel_unit = parallel_unit;
  loop->numChunks = numChunks;
  return loop;
}

// Spatial Memory Load
Stmt MemLoad::make(Expr lhsMem, Expr rhsMem, Expr start, Expr offset, Expr par) {
  MemLoad *memLoad = new MemLoad;
  memLoad->lhsMem = lhsMem;
  memLoad->rhsMem = rhsMem;
  memLoad->start = start;
  memLoad->offset = offset;
  memLoad->par = par;
  return memLoad;
}

// Spatial Memory Store
Stmt MemStore::make(Expr lhsMem, Expr rhsMem, Expr start, Expr offset, Expr par) {
  MemStore *memStore = new MemStore;
  memStore->lhsMem = lhsMem;
  memStore->rhsMem = rhsMem;
  memStore->start = start;
  memStore->offset = offset;
  memStore->par = par;
  return memStore;
}

Stmt GenBitVector::make(Expr shift, Expr out_bitcnt, Expr in_len, Expr in_fifo, Expr out_fifo) {
  GenBitVector *genBitVector = new GenBitVector;
  genBitVector->shift = shift;
  genBitVector->out_bitcnt = out_bitcnt;
  genBitVector->in_len = in_len;
  genBitVector->in_fifo = in_fifo;
  genBitVector->out_fifo = out_fifo;
  return genBitVector;
}

Expr Scan::make(Expr par, Expr bitcnt, Expr in_fifo1, Expr in_fifo2, bool or_op, bool reduction) {
  Scan *scan = new Scan;
  scan->par = par;
  scan->bitcnt = bitcnt;
  scan->in_fifo1 = in_fifo1;
  scan->in_fifo2 = in_fifo2;
  scan->or_op = or_op;
  scan->reduction = reduction;
  return scan;
}

Expr TypeCase::make(std::vector<ir::Expr> vars) {
  TypeCase *typeCase = new TypeCase;
  typeCase->vars = vars;
  return typeCase;
}

Expr RMW::make(Expr arr, Expr addr, Expr data, Expr barrier, SpatialRMWoperators op, SpatialMemOrdering ordering) {
  RMW *rmw = new RMW;
  rmw->arr = arr;
  rmw->addr = addr;
  rmw->data = data;
  rmw->barrier = barrier;
  rmw->op = op;
  rmw->ordering = ordering;
  return rmw;
}

Stmt CallStmt::make(Expr call) {
  CallStmt *callStmt = new CallStmt;
  callStmt->call = call;
  return callStmt;
}

/// SPATIAL ONLY END
  
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
template<> void ExprNode<BitOr>::accept(IRVisitorStrict *v)
    const { v->visit((const BitOr*)this); }
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
template<> void ExprNode<Cast>::accept(IRVisitorStrict *v)
    const { v->visit((const Cast*)this); }
template<> void ExprNode<Call>::accept(IRVisitorStrict *v)
    const { v->visit((const Call*)this); }
template<> void StmtNode<CallStmt>::accept(IRVisitorStrict *v)
    const { v->visit((const CallStmt*)this); }
template<> void StmtNode<IfThenElse>::accept(IRVisitorStrict *v)
    const { v->visit((const IfThenElse*)this); }
template<> void ExprNode<Ternary>::accept(IRVisitorStrict *v)
    const { v->visit((const Ternary*)this); }
template<> void StmtNode<Case>::accept(IRVisitorStrict *v)
    const { v->visit((const Case*)this); }
template<> void StmtNode<Switch>::accept(IRVisitorStrict *v)
    const { v->visit((const Switch*)this); }
template<> void ExprNode<Load>::accept(IRVisitorStrict *v)
    const { v->visit((const Load*)this); }
template<> void ExprNode<Malloc>::accept(IRVisitorStrict *v)
    const { v->visit((const Malloc*)this); }
template<> void ExprNode<Sizeof>::accept(IRVisitorStrict *v)
    const { v->visit((const Sizeof*)this); }
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
template<> void StmtNode<VarDecl>::accept(IRVisitorStrict *v)
    const { v->visit((const VarDecl*)this); }
template<> void StmtNode<Assign>::accept(IRVisitorStrict *v)
    const { v->visit((const Assign*)this); }
template<> void StmtNode<Yield>::accept(IRVisitorStrict *v)
    const { v->visit((const Yield*)this); }
template<> void StmtNode<Allocate>::accept(IRVisitorStrict *v)
    const { v->visit((const Allocate*)this); }
template<> void StmtNode<Free>::accept(IRVisitorStrict *v)
    const { v->visit((const Free*)this); }
template<> void StmtNode<Comment>::accept(IRVisitorStrict *v)
    const { v->visit((const Comment*)this); }
template<> void StmtNode<BlankLine>::accept(IRVisitorStrict *v)
    const { v->visit((const BlankLine*)this); }
template<> void StmtNode<Continue>::accept(IRVisitorStrict *v)
  const { v->visit((const Continue*)this); }
template<> void StmtNode<Print>::accept(IRVisitorStrict *v)
    const { v->visit((const Print*)this); }
template<> void ExprNode<GetProperty>::accept(IRVisitorStrict *v)
    const { v->visit((const GetProperty*)this); }
template<> void StmtNode<Sort>::accept(IRVisitorStrict *v)
  const { v->visit((const Sort*)this); }
template<> void StmtNode<Break>::accept(IRVisitorStrict *v)
  const { v->visit((const Break*)this); }
template<> void StmtNode<StoreBulk>::accept(IRVisitorStrict *v)
    const { v->visit((const StoreBulk*)this); }
template<> void ExprNode<LoadBulk>::accept(IRVisitorStrict *v)
    const { v->visit((const LoadBulk*)this); }
/// SPATIAL ONLY
template<> void StmtNode<Reduce>::accept(IRVisitorStrict *v)
    const { v->visit((const Reduce*)this); }
template<> void StmtNode<ReduceScan>::accept(IRVisitorStrict *v)
    const { v->visit((const ReduceScan*)this); }
template<> void StmtNode<ForScan>::accept(IRVisitorStrict *v)
    const { v->visit((const ForScan*)this); }
template<> void StmtNode<MemLoad>::accept(IRVisitorStrict *v)
    const { v->visit((const MemLoad*)this); }
template<> void StmtNode<MemStore>::accept(IRVisitorStrict *v)
    const { v->visit((const MemStore*)this); }
template<> void StmtNode<GenBitVector>::accept(IRVisitorStrict *v)
    const { v->visit((const GenBitVector*)this); }
template<> void ExprNode<Scan>::accept(IRVisitorStrict *v)
    const { v->visit((const Scan*)this); }
template<> void ExprNode<TypeCase>::accept(IRVisitorStrict *v)
    const { v->visit((const TypeCase*)this); }
template<> void ExprNode<RMW>::accept(IRVisitorStrict *v)
    const { v->visit((const RMW*)this); }


// printing methods
std::ostream& operator<<(std::ostream& os, const Stmt& stmt) {
  if (!stmt.defined()) return os << "Stmt()" << std::endl;
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

Stmt rewriteBulkStmt(Stmt stmt, IndexVar indexVar) {
  struct BulkRewriteStmt : IRRewriter {
    using IRRewriter::visit;
    IndexVar i;
    BulkRewriteStmt(IndexVar indexVar) : i(indexVar) {}

    void visit(const Block* node) {
      if (node->contents.size() > 1) {
        std::vector<Stmt> rewritten;
        for (auto it = node->contents.begin(); it != node->contents.end(); it++) {
          rewritten.push_back(*it);
        }
        stmt = Block::make(rewritten);
      }
      IRRewriter::visit(node);
    }

    void visit(const Var* node) {
      if (node->name == i.getName()) {
        expr = Literal::make(0);
      }
      else {
        expr = node;
      }
    }
  };
  return BulkRewriteStmt(indexVar).rewrite(stmt);
}

Expr rewriteBulkExpr(Stmt stmt, IndexVar indexVar) {

  struct BulkExprFinder : IRVisitor {
    using IRVisitor::visit;
    BulkExprFinder() = default;
    Stmt parentStmt;
    Expr e = Expr();
    Expr visit(Stmt stmt) {
      if (stmt.defined())
        stmt.accept(this);
      return e;
    }
    void visit(const Block* node) {
      parentStmt = node;
      node->contents.back().accept(this);
    }
    void visit(const Assign* node) {
      if (parentStmt.defined())
        e = node->rhs;
    }
    void visit(const VarDecl* node) {
      if (parentStmt.defined())
        e = node->rhs;
    }
  };
  struct BulkExprRewriter : IRRewriter {
    using IRRewriter::visit;
    IndexVar i;
    BulkExprRewriter(IndexVar indexVar) : i(indexVar) {}

    void visit(const Var* node) {
      if (node->name == i.getName()) {
        expr = Literal::make(0);
      }
      else {
        expr = node;
      }
    }
  };

  Expr rewrittenExpr = BulkExprFinder().visit(stmt);
  Expr returnExpr = BulkExprRewriter(indexVar).rewrite(rewrittenExpr);
  return returnExpr;
}

Stmt rewriteStmtRemoveDuplicates(Stmt stmt1, Stmt stmt2) {
  struct GetDecls : IRRewriter {
    using IRRewriter::visit;
    std::map<Expr,Stmt> declarations;

    GetDecls() = default;

    void visit(const VarDecl* decl) {
      Expr rhs = rewrite(decl->rhs);
      stmt = (rhs == decl->rhs) ? decl : VarDecl::make(decl->var, rhs);

      declarations.insert({decl->var, stmt});
    }
  };

  struct RemoveDuplicates : IRRewriter {
    using IRRewriter::visit;
    std::map<Expr,Stmt> declarations;

    RemoveDuplicates(std::map<Expr,Stmt> decls) : declarations(decls) {}

    void visit(const VarDecl* decl) {
      Expr rhs = rewrite(decl->rhs);
      stmt = (rhs == decl->rhs) ? decl : VarDecl::make(decl->var, rhs);
      if (declarations.find(decl->var) != declarations.end())
        stmt = Stmt();
      else
        declarations.insert({decl->var, stmt});
    }
  };

  auto decls = GetDecls();
  decls.rewrite(stmt1);

  auto simplifiedStmt = RemoveDuplicates(decls.declarations).rewrite(stmt2);
  return simplifiedStmt;
}

} // namespace ir
} // namespace taco
