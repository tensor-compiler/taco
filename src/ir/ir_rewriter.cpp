#include "taco/ir/ir_rewriter.h"

#include <vector>

#include "taco/ir/ir.h"

using namespace std;

namespace taco {
namespace ir {

// class IRRewriter
IRRewriter::~IRRewriter() {
}

Expr IRRewriter::rewrite(Expr e) {
  if (e.defined()) {
    e.accept(this);
    e = expr;
  }
  else {
    e = Expr();
  }
  expr = Expr();
  stmt = Stmt();
  return e;
}

Stmt IRRewriter::rewrite(Stmt s) {
  if (s.defined()) {
    s.accept(this);
    s = stmt;
  }
  else {
    s = Stmt();
  }
  expr = Expr();
  stmt = Stmt();
  return s;
}

template <class T>
Expr visitUnaryOp(const T *op, IRRewriter *rw) {
  Expr a = rw->rewrite(op->a);
  if (a == op->a) {
    return op;
  }
  else {
    return T::make(a);
  }
}

template <class T>
Expr visitBinaryOp(const T *op, IRRewriter *rw) {
  Expr a = rw->rewrite(op->a);
  Expr b = rw->rewrite(op->b);
  if (a == op->a && b == op->b) {
    return op;
  }
  else {
    return T::make(a, b);
  }
}

void IRRewriter::visit(const Literal* op) {
  expr = op;
}

void IRRewriter::visit(const Var* op) {
  expr = op;
}

void IRRewriter::visit(const Neg* op) {
  expr = visitUnaryOp(op, this);
}

void IRRewriter::visit(const Sqrt* op) {
  expr = visitUnaryOp(op, this);
}

void IRRewriter::visit(const Add* op) {
  expr = visitBinaryOp(op, this);
}

void IRRewriter::visit(const Sub* op) {
  expr = visitBinaryOp(op, this);
}

void IRRewriter::visit(const Mul* op) {
  expr = visitBinaryOp(op, this);
}

void IRRewriter::visit(const Div* op) {
  expr = visitBinaryOp(op, this);
}

void IRRewriter::visit(const Rem* op) {
  expr = visitBinaryOp(op, this);
}

void IRRewriter::visit(const Min* op) {
  vector<Expr> operands;
  bool operandsSame = true;
  for (const Expr& operand : op->operands) {
    Expr rewrittenOperand = rewrite(operand);
    operands.push_back(rewrittenOperand);
    if (rewrittenOperand != operand) {
      operandsSame = false;
    }
  }
  if (operandsSame) {
    expr = op;
  }
  else {
    expr = Min::make(operands);
  }
}

void IRRewriter::visit(const Max* op) {
  vector<Expr> operands;
  bool operandsSame = true;
  for (const Expr& operand : op->operands) {
    Expr rewrittenOperand = rewrite(operand);
    operands.push_back(rewrittenOperand);
    if (rewrittenOperand != operand) {
      operandsSame = false;
    }
  }
  if (operandsSame) {
    expr = op;
  }
  else {
    expr = Max::make(operands);
  }
}

void IRRewriter::visit(const BitAnd* op) {
  expr = visitBinaryOp(op, this);
}

void IRRewriter::visit(const BitOr* op) {
  expr = visitBinaryOp(op, this);
}

void IRRewriter::visit(const Eq* op) {
  expr = visitBinaryOp(op, this);
}

void IRRewriter::visit(const Neq* op) {
  expr = visitBinaryOp(op, this);
}

void IRRewriter::visit(const Gt* op) {
  expr = visitBinaryOp(op, this);
}

void IRRewriter::visit(const Lt* op) {
  expr = visitBinaryOp(op, this);
}

void IRRewriter::visit(const Gte* op) {
  expr = visitBinaryOp(op, this);
}

void IRRewriter::visit(const Lte* op) {
  expr = visitBinaryOp(op, this);
}

void IRRewriter::visit(const And* op) {
  expr = visitBinaryOp(op, this);
}

void IRRewriter::visit(const Or* op) {
  expr = visitBinaryOp(op, this);
}

void IRRewriter::visit(const Cast* op) {
  Expr a = rewrite(op->a);
  if (a == op->a) {
    expr = op;
  }
  else {
    expr = Cast::make(a, op->type);
  }
}

void IRRewriter::visit(const Call* op) {
  std::vector<Expr> args;
  bool rewritten = false;
  for (auto& arg : op->args) {
    Expr rewrittenArg = rewrite(arg);
    args.push_back(rewrittenArg);
    if (rewrittenArg != arg) {
      rewritten = true;
    }
  }
  if (rewritten) {
    expr = Call::make(op->func, args, op->type);
  }
  else {
    expr = op;
  }
}

void IRRewriter::visit(const CallStmt* op) {
  Expr call = rewrite(op->call);
  if (call == op->call) {
    stmt = op;
  } else {
    stmt = ir::CallStmt::make(call);
  }
}

void IRRewriter::visit(const IfThenElse* op) {
  Expr cond      = rewrite(op->cond);
  Stmt then      = rewrite(op->then);
  Stmt otherwise = rewrite(op->otherwise);
  if (cond == op->cond && then == op->then && otherwise == op->otherwise) {
    stmt = op;
  }
  else {
    stmt = otherwise.defined() ? IfThenElse::make(cond, then, otherwise)
                               : IfThenElse::make(cond, then);
  }
}

void IRRewriter::visit(const Ternary* op) {
  Expr cond      = rewrite(op->cond);
  Expr then      = rewrite(op->then);
  Expr otherwise = rewrite(op->otherwise);
  if (cond == op->cond && then == op->then && otherwise == op->otherwise) {
    expr = op;
  }
  else {
    expr = Ternary::make(cond, then, otherwise);
  }
}


void IRRewriter::visit(const Case* op) {
  vector<std::pair<Expr,Stmt>> clauses;
  bool clausesSame = true;
  for (auto& clause : op->clauses) {
    Expr clauseExpr = rewrite(clause.first);
    Stmt clauseStmt = rewrite(clause.second);
    clauses.push_back({clauseExpr, clauseStmt});
    if (clauseExpr != clause.first || clauseStmt != clause.second) {
      clausesSame = false;
    }
  }
  if (clausesSame) {
    stmt = op;
  }
  else {
    stmt = Case::make(clauses, op->alwaysMatch);
  }
}

void IRRewriter::visit(const Switch* op) {
  Expr controlExpr = rewrite(op->controlExpr);
  vector<std::pair<Expr,Stmt>> cases;
  bool casesSame = true;
  for (auto& switchCase : op->cases) {
    Expr caseExpr = rewrite(switchCase.first);
    Stmt caseStmt = rewrite(switchCase.second);
    cases.push_back({caseExpr, caseStmt});
    if (caseExpr != switchCase.first || caseStmt != switchCase.second) {
      casesSame = false;
    }
  }
  if (controlExpr == op->controlExpr && casesSame) {
    stmt = op;
  }
  else {
    stmt = Switch::make(cases, controlExpr);
  }
}

void IRRewriter::visit(const Load* op) {
  Expr arr = rewrite(op->arr);
  Expr loc = rewrite(op->loc);
  if (arr == op->arr && loc == op->loc) {
    expr = op;
  }
  else {
    expr = loc.defined() ? Load::make(arr, loc, op->mem_loc) : Load::make(arr);
  }
}

void IRRewriter::visit(const Malloc* op) {
  Expr size = rewrite(op->size);
  if (size == op->size) {
    expr = op;
  }
  else {
    expr = Malloc::make(size);
  }
}

void IRRewriter::visit(const Sizeof* op) {
  expr = op;
}

void IRRewriter::visit(const Store* op) {
  Expr arr  = rewrite(op->arr);
  Expr loc  = rewrite(op->loc);
  Expr data = rewrite(op->data);
  if (arr == op->arr && loc == op->loc && data == op->data) {
    stmt = op;
  }
  else {
    stmt = Store::make(arr, loc, data, op->lhs_mem_loc, op->rhs_mem_loc, op->use_atomics);
  }
}

void IRRewriter::visit(const For* op) {
  Expr var       = rewrite(op->var);
  Expr start     = rewrite(op->start);
  Expr end       = rewrite(op->end);
  Expr increment = rewrite(op->increment);
  Expr numChunks = rewrite(op->numChunks);
  Stmt contents  = rewrite(op->contents);

  if (var == op->var && start == op->start && end == op->end &&
      increment == op->increment && contents == op->contents && numChunks == op->numChunks) {
    stmt = op;
  }
  else {
    stmt = For::make(var, start, end, increment, numChunks, contents, op->kind,
                     op->parallel_unit, op->unrollFactor, op->vec_width);
  }
}

void IRRewriter::visit(const While* op) {
  Expr cond     = rewrite(op->cond);
  Stmt contents = rewrite(op->contents);
  if (cond == op->cond && contents == op->contents) {
    stmt = op;
  }
  else {
    stmt = While::make(cond, contents, op->kind, op->vec_width);
  }
}

void IRRewriter::visit(const Block* op) {
  vector<Stmt> contents;
  bool contentsSame = true;
  for (auto& content : op->contents) {
    Stmt rewrittenContent = rewrite(content);

    if (rewrittenContent.defined()) {
      contents.push_back(rewrittenContent);
    }

    if (rewrittenContent != content) {
      contentsSame = false;
    }
  }
  if (contentsSame) {
    stmt = op;
  }
  else {
    stmt = Block::make(contents);
  }
}

void IRRewriter::visit(const Scope* op) {
  Stmt scopedStmt = rewrite(op->scopedStmt);

  Expr returnExpr;
  if (op->returnExpr.defined()) {
    returnExpr = rewrite(op->returnExpr);
  }

  if ((scopedStmt == op->scopedStmt && !returnExpr.defined()) ||
      (scopedStmt == op->scopedStmt && returnExpr.defined() && returnExpr == op->returnExpr)) {
    stmt = op;
  }
  else if (returnExpr.defined()) {
    stmt = Scope::make(scopedStmt, returnExpr);
  }
  else {
    stmt = Scope::make(scopedStmt);
  }
}

void IRRewriter::visit(const Function* op) {
  Stmt body = rewrite(op->body);
  vector<Expr> inputs;
  vector<Expr> outputs;
  bool inputOutputsSame = true;
  for (auto& input : op->inputs) {
    Expr rewrittenInput = rewrite(input);
    inputs.push_back(rewrittenInput);
    if (rewrittenInput != input) {
      inputOutputsSame = false;
    }
  }
  for (auto& output : op->outputs) {
    Expr rewrittenOutput = rewrite(output);
    outputs.push_back(rewrittenOutput);
    if (rewrittenOutput != output) {
      inputOutputsSame = false;
    }
  }
  if (body == op->body && inputOutputsSame) {
    stmt = op;
  }
  else {
    stmt = Function::make(op->name, outputs, inputs, body);
  }
}

void IRRewriter::visit(const VarDecl* op) {
  Expr var = rewrite(op->var);
  Expr rhs = rewrite(op->rhs);
  if (var == op->var && rhs == op->rhs) {
    stmt = op;
  }
  else {
    stmt = VarDecl::make(var, rhs, op->mem);
  }
}

void IRRewriter::visit(const Assign* op) {
  Expr lhs = rewrite(op->lhs);
  Expr rhs = rewrite(op->rhs);
  if (lhs == op->lhs && rhs == op->rhs) {
    stmt = op;
  }
  else {
    stmt = Assign::make(lhs, rhs, op->use_atomics, op->atomic_parallel_unit);
  }
}

void IRRewriter::visit(const Yield* op) {
  std::vector<Expr> coords;
  bool coordsSame = true;
  for (auto& coord : op->coords) {
    Expr rewrittenCoord = rewrite(coord);
    coords.push_back(coord);
    if (rewrittenCoord != coord) {
      coordsSame = false;
    }
  }
  Expr val = rewrite(op->val);
  if (val == op->val && coordsSame) {
    stmt = op;
  }
  else {
    stmt = Yield::make(coords, val);
  }
}

void IRRewriter::visit(const Allocate* op) {
  Expr var          = rewrite(op->var);
  Expr num_elements = rewrite(op->num_elements);
  if (var == op->var && num_elements == op->num_elements) {
    stmt = op;
  }
  else {
    stmt = Allocate::make(var, num_elements, op->is_realloc, op->old_elements, op->clear, op->memoryLocation);
  }
}

void IRRewriter::visit(const Free* op) {
  Expr var = rewrite(op->var);
  if (var == op->var) {
    stmt = op;
  }
  else {
    stmt = Free::make(var);
  }
}

void IRRewriter::visit(const Comment* op) {
  stmt = op;
}

void IRRewriter::visit(const BlankLine* op) {
  stmt = op;
}

void IRRewriter::visit(const Continue* op) {
  stmt = op;
}

void IRRewriter::visit(const Break* op) {
  stmt = op;
}

void IRRewriter::visit(const Print* op) {
  vector<Expr> params;
  bool paramsSame = true;
  for (auto& param : op->params) {
    Expr rewrittenParam = rewrite(param);
    params.push_back(rewrittenParam);
    if (rewrittenParam != param) {
      paramsSame = false;
    }
  }
  if (paramsSame) {
    stmt = op;
  }
  else {
    stmt = Print::make(op->fmt, params);
  }
}

void IRRewriter::visit(const GetProperty* op) {
  Expr tensor = rewrite(op->tensor);
  if (tensor == op->tensor) {
    expr = op;
  }
  else {
    expr = GetProperty::make(tensor, op->property, op->mode, op->index, op->name);
  }
}

void IRRewriter::visit(const Sort* op) {
  std::vector<Expr> args;
  bool rewritten = false;
  for (auto& arg : op->args) {
    Expr rewrittenArg = rewrite(arg);
    args.push_back(rewrittenArg);
    if (rewrittenArg != arg) {
      rewritten = true;
    }
  }
  if (rewritten) {
    stmt = Sort::make(args);
  }
  else {
    stmt = op;
  }
}

void IRRewriter::visit(const StoreBulk* op) {
  Expr arr      = rewrite(op->arr);
  Expr locStart = Expr();
  Expr locEnd   = Expr();
  if (op->locStart.defined() && op->locEnd.defined()) {
    locStart = rewrite(op->locStart);
    locEnd = rewrite(op->locEnd);
  }
  Expr data     = rewrite(op->data);

  if (arr == op->arr && (!op->locStart.defined() || (op->locStart.defined() && locStart == op->locStart)) &&
    (!op->locEnd.defined() || (op->locEnd.defined() && locEnd == op->locEnd)) && data == op->data) {
    stmt = op;
  }
  else {
    stmt = StoreBulk::make(arr, locStart, locEnd, data, op->lhs_mem_loc, op->rhs_mem_loc, op->use_atomics, op->atomic_parallel_unit);
  }
}

void IRRewriter::visit(const LoadBulk* op) {
  Expr arr      = rewrite(op->arr);
  Expr locStart = rewrite(op->locStart);
  Expr locEnd   = rewrite(op->locEnd);
  if (arr == op->arr && locStart == op->locStart && locEnd == op->locEnd) {
    expr = op;
  }
  else {
    expr = LoadBulk::make(arr, locStart, locEnd);
  }
}

/// SPATIAL ONLY
void IRRewriter::visit(const Reduce* op) {
  Expr var       = rewrite(op->var);
  Expr reg       = rewrite(op->reg);
  Expr start     = rewrite(op->start);
  Expr end       = rewrite(op->end);
  Expr increment = rewrite(op->increment);
  Expr numChunks = rewrite(op->numChunks);
  Stmt contents  = rewrite(op->contents);
  Expr retExpr;
  if (op->returnExpr.defined())
    retExpr   = rewrite(op->returnExpr);

  if (var == op->var && reg == op->reg && start == op->start && end == op->end &&
      increment == op->increment && contents == op->contents && (!op->returnExpr.defined() || retExpr == op->returnExpr)
      && numChunks == op->numChunks) {
    stmt = op;
  } else {
    stmt = Reduce::make(var, reg, start, end, increment, numChunks, contents, retExpr, op->add);
  }
}

void IRRewriter::visit(const ReduceScan* op) {
  Expr caseType  = rewrite(op->caseType);
  Expr reg       = rewrite(op->reg);
  Expr scanner   = rewrite(op->scanner);
  Stmt contents;
  if (op->contents.defined())
    contents  = rewrite(op->contents);
  Expr retExpr;
  if (op->returnExpr.defined())
    retExpr   = rewrite(op->returnExpr);

  if (caseType == op->caseType && reg == op->reg && scanner == op->scanner && contents == op->contents &&
      !op->returnExpr.defined() && !op->contents.defined()) {
    stmt = op;
  } else if (caseType == op->caseType && reg == op->reg && scanner == op->scanner && !op->contents.defined()
             && op->returnExpr.defined() && retExpr == op->returnExpr) {
    stmt = op;
  } else if (caseType == op->caseType && reg == op->reg && scanner == op->scanner && op->contents.defined() && contents == op->contents
             && op->returnExpr.defined() && retExpr == op->returnExpr ) {
    stmt = op;
  } else if (caseType == op->caseType && reg == op->reg && scanner == op->scanner && op->contents.defined() && contents == op->contents
             && !op->returnExpr.defined()) {
    stmt = op;
  } else {
    stmt = ReduceScan::make(caseType, reg, scanner, contents, retExpr, op->add);
  }
}

void IRRewriter::visit(const ForScan* op) {
  Expr caseType  = rewrite(op->caseType);
  Expr scanner   = rewrite(op->scanner);
  Stmt contents  = rewrite(op->contents);
  if (caseType == op->caseType && scanner == op->scanner && contents == op->contents) {
    stmt = op;
  }
  else {
    stmt = ForScan::make(caseType, scanner, contents, op->kind,
                     op->parallel_unit, op->unrollFactor, op->vec_width, op->numChunks);
  }
}

void IRRewriter::visit(const MemLoad* op) {
  Expr lhsMem       = rewrite(op->lhsMem);
  Expr rhsMem       = rewrite(op->rhsMem);
  Expr start        = rewrite(op->start);
  Expr offset       = rewrite(op->offset);
  Expr par          = rewrite(op->par);
  if (lhsMem == op->lhsMem && rhsMem == op->rhsMem && start == op->start && offset == op->offset && par == op->par) {
    stmt = op;
  }
  else {
    stmt = MemLoad::make(lhsMem, rhsMem, start, offset, par);
  }
}

void IRRewriter::visit(const MemStore* op) {
  Expr lhsMem       = rewrite(op->lhsMem);
  Expr rhsMem       = rewrite(op->rhsMem);
  Expr start        = rewrite(op->start);
  Expr offset       = rewrite(op->offset);
  Expr par          = rewrite(op->par);
  if (lhsMem == op->lhsMem && rhsMem == op->rhsMem && start == op->start && offset == op->offset && par == op->par) {
    stmt = op;
  }
  else {
    stmt = MemStore::make(lhsMem, rhsMem, start, offset, par);
  }
}
void IRRewriter::visit(const GenBitVector* op) {
  Expr shift        = rewrite(op->shift);
  Expr out_bitcnt   = rewrite(op->out_bitcnt);
  Expr in_len       = rewrite(op->in_len);
  Expr in_fifo      = rewrite(op->in_fifo);
  Expr out_fifo     = rewrite(op->out_fifo);

  if (shift == op->shift && out_bitcnt == op->out_bitcnt && in_len == op->in_len && in_fifo == op->in_fifo
      && out_fifo == op->out_fifo) {
    stmt = op;
  } else {
    stmt = GenBitVector::make(shift, out_bitcnt, in_len, in_fifo, out_fifo);
  }
}

void IRRewriter::visit(const Scan* op) {
  Expr par          = rewrite(op->par);
  Expr bitcnt       = rewrite(op->bitcnt);
  Expr in_fifo1     = rewrite(op->in_fifo1);
  Expr in_fifo2;
  if (op->in_fifo2.defined()) {
    in_fifo2        = rewrite(op->in_fifo2);
  }

  if (par == op->par && bitcnt == op->bitcnt && in_fifo1 == op->in_fifo1 &&
      (!op->in_fifo2.defined() || in_fifo2 == op->in_fifo2)) {
    expr = op;
  } else if (op->in_fifo2.defined()){
    expr = Scan::make(par, bitcnt, in_fifo1, in_fifo2, op->or_op, op->reduction);
  } else {
    expr = Scan::make(par, bitcnt, in_fifo1, op->or_op, op->reduction);
  }
}

void IRRewriter::visit(const TypeCase* op) {
  vector<Expr> vars;
  bool varsSame = true;
  for (auto& var : op->vars) {
    Expr rewrittenVar = rewrite(var);

    if (rewrittenVar.defined()) {
      vars.push_back(rewrittenVar);
    }

    if (rewrittenVar != var) {
      varsSame = false;
    }
  }
  if (varsSame) {
    expr = op;
  }
  else {
    expr = TypeCase::make(vars);
  }
}

void IRRewriter::visit(const RMW* op) {
  Expr arr      = rewrite(op->arr);
  Expr addr     = rewrite(op->addr);
  Expr data     = rewrite(op->data);
  Expr barrier;
  if (op->barrier.defined())
    barrier  = rewrite(op->barrier);

  if (arr == op->arr && addr == op->addr && data == op->data && (!op->barrier.defined() || barrier == op->barrier)) {
    expr = op;
  } else {
    expr = ir::RMW::make(arr, addr, data, barrier, op->op, op->ordering);
  }
}

void IRRewriter::visit(const FuncEnv* op) {
  Stmt env      = rewrite(op->env);

  if (env == op->env ) {
    stmt = op;
  } else {
    stmt = ir::FuncEnv::make(env);
  }
}

void IRRewriter::visit(const AccelEnv* op) {
  Stmt aenv      = rewrite(op->aenv);

  if (aenv == op->aenv ) {
    stmt = op;
  } else {
    stmt = ir::FuncEnv::make(aenv);
  }
}
/// SPATIAL ONLY END

}}
