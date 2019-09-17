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
  expr = visitBinaryOp(op, this);
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
    expr = loc.defined() ? Load::make(arr, loc) : Load::make(arr);
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
    stmt = Store::make(arr, loc, data);
  }
}

void IRRewriter::visit(const For* op) {
  Expr var       = rewrite(op->var);
  Expr start     = rewrite(op->start);
  Expr end       = rewrite(op->end);
  Expr increment = rewrite(op->increment);
  Stmt contents  = rewrite(op->contents);
  if (var == op->var && start == op->start && end == op->end &&
      increment == op->increment && contents == op->contents) {
    stmt = op;
  }
  else {
    stmt = For::make(var, start, end, increment, contents, op->kind,
                     op->accelerator, op->vec_width);
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
  if (scopedStmt == op->scopedStmt) {
    stmt = op;
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
    stmt = VarDecl::make(var, rhs);
  }
}

void IRRewriter::visit(const Assign* op) {
  Expr lhs = rewrite(op->lhs);
  Expr rhs = rewrite(op->rhs);
  if (lhs == op->lhs && rhs == op->rhs) {
    stmt = op;
  }
  else {
    stmt = Assign::make(lhs, rhs);
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
    stmt = Allocate::make(var, num_elements, op->is_realloc, op->old_elements);
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


}}
