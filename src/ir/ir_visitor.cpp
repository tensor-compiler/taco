#include "taco/ir/ir_visitor.h"

#include "taco/ir/ir.h"

namespace taco {
namespace ir {

// class IRVisitorStrict
IRVisitorStrict::~IRVisitorStrict() {
}


// class IRVisitor
IRVisitor::~IRVisitor() {
}

void IRVisitor::visit(const Literal*) {
}

void IRVisitor::visit(const Var*) {
}

void IRVisitor::visit(const Neg* op) {
  op->a.accept(this);
}

void IRVisitor::visit(const Sqrt* op) {
  op->a.accept(this);
}

void IRVisitor::visit(const Add* op) {
  op->a.accept(this);
  op->b.accept(this);
}

void IRVisitor::visit(const Sub* op){
  op->a.accept(this);
  op->b.accept(this);
}

void IRVisitor::visit(const Mul* op){
  op->a.accept(this);
  op->b.accept(this);
}

void IRVisitor::visit(const Div* op){
  op->a.accept(this);
  op->b.accept(this);
}

void IRVisitor::visit(const Rem* op){
  op->a.accept(this);
  op->b.accept(this);
}

void IRVisitor::visit(const Min* op){
  for (auto e: op->operands) {
    e.accept(this);
  }
}

void IRVisitor::visit(const Max* op){
  for (auto e: op->operands) {
    e.accept(this);
  }
}

void IRVisitor::visit(const BitAnd* op){
  op->a.accept(this);
  op->b.accept(this);
}

void IRVisitor::visit(const BitOr* op){
  op->a.accept(this);
  op->b.accept(this);
}

void IRVisitor::visit(const Eq* op){
  op->a.accept(this);
  op->b.accept(this);
}

void IRVisitor::visit(const Neq* op){
  op->a.accept(this);
  op->b.accept(this);
}

void IRVisitor::visit(const Gt* op){
  op->a.accept(this);
  op->b.accept(this);
}

void IRVisitor::visit(const Lt* op){
  op->a.accept(this);
  op->b.accept(this);
}

void IRVisitor::visit(const Gte* op){
  op->a.accept(this);
  op->b.accept(this);
}

void IRVisitor::visit(const Lte* op){
  op->a.accept(this);
  op->b.accept(this);
}

void IRVisitor::visit(const And* op){
  op->a.accept(this);
  op->b.accept(this);
}

void IRVisitor::visit(const Or* op){
  op->a.accept(this);
  op->b.accept(this);
}

void IRVisitor::visit(const Cast* op){
  op->a.accept(this);
}

void IRVisitor::visit(const Call* op) {
  for (auto& arg : op->args) {
    arg.accept(this);
  }
}

void IRVisitor::visit(const CallStmt* op) {
  op->call.accept(this);
}

void IRVisitor::visit(const IfThenElse* op) {
  op->cond.accept(this);
  op->then.accept(this);

  if (op->otherwise.defined()) {
    op->otherwise.accept(this);
  }
}

void IRVisitor::visit(const Ternary* op) {
  op->cond.accept(this);
  op->then.accept(this);
  op->otherwise.accept(this);
}

void IRVisitor::visit(const Case* op) {
  for (auto clause : op->clauses) {
    clause.first.accept(this);
    clause.second.accept(this);
  }
}

void IRVisitor::visit(const Switch* op) {
  op->controlExpr.accept(this);
  for (auto switchCase : op->cases) {
    switchCase.first.accept(this);
    switchCase.second.accept(this);
  }
}

void IRVisitor::visit(const Load* op) {
  op->arr.accept(this);
  op->loc.accept(this);
}

void IRVisitor::visit(const Malloc* op) {
  op->size.accept(this);
}

void IRVisitor::visit(const Sizeof* op) {
}

void IRVisitor::visit(const Store* op) {
  op->arr.accept(this);
  op->loc.accept(this);
  op->data.accept(this);
}

void IRVisitor::visit(const For* op) {
  op->var.accept(this);
  op->start.accept(this);
  op->end.accept(this);
  op->increment.accept(this);
  op->numChunks.accept(this);
  op->contents.accept(this);
}

void IRVisitor::visit(const While* op) {
  op->cond.accept(this);
  op->contents.accept(this);
}

void IRVisitor::visit(const Block* op) {
  for (auto s:op->contents)
    s.accept(this);
}

void IRVisitor::visit(const Scope* op) {
  op->scopedStmt.accept(this);
}

void IRVisitor::visit(const Function* op) {
  if (op->funcEnv.defined())
    op->funcEnv.accept(this);
  if (op->accelEnv.defined())
    op->accelEnv.accept(this);

    op->body.accept(this);
}

void IRVisitor::visit(const VarDecl* op) {
  op->rhs.accept(this);
}

void IRVisitor::visit(const Assign* op) {
  op->lhs.accept(this);
  op->rhs.accept(this);
}

void IRVisitor::visit(const Yield* op) {
  for (auto coord : op->coords) {
    coord.accept(this);
  }
  op->val.accept(this);
}

void IRVisitor::visit(const Allocate* op) {
  op->var.accept(this);
  op->num_elements.accept(this);
}

void IRVisitor::visit(const Free* op) {
  op->var.accept(this);
}

void IRVisitor::visit(const GetProperty* op) {
  op->tensor.accept(this);
}

void IRVisitor::visit(const Comment*) {
}

void IRVisitor::visit(const BlankLine*) {
}

void IRVisitor::visit(const Continue*) {
}

void IRVisitor::visit(const Break*) {
}

void IRVisitor::visit(const Print* op) {
  for (auto e: op->params)
    e.accept(this);
}

void IRVisitor::visit(const Sort* op) {
  for (auto e: op->args)
    e.accept(this);
}

void IRVisitor::visit(const LoadBulk* op) {
  op->arr.accept(this);
  op->locStart.accept(this);
  op->locEnd.accept(this);
}

void IRVisitor::visit(const StoreBulk* op) {
  op->arr.accept(this);
  if (op->locStart.defined() && op->locEnd.defined()) {
    op->locStart.accept(this);
    op->locEnd.accept(this);
  }
  op->data.accept(this);
}

/// SPATIAL ONLY
void IRVisitor::visit(const Reduce* op) {
  op->var.accept(this);
  op->reg.accept(this);
  op->start.accept(this);
  op->end.accept(this);
  op->increment.accept(this);
  op->numChunks.accept(this);
  op->contents.accept(this);
  if (op->returnExpr.defined())
    op->returnExpr.accept(this);
}

void IRVisitor::visit(const ReduceScan* op) {
  op->caseType.accept(this);
  op->reg.accept(this);
  op->scanner.accept(this);
  if (op->contents.defined())
    op->contents.accept(this);
  if (op->returnExpr.defined())
    op->returnExpr.accept(this);
}

void IRVisitor::visit(const ForScan* op) {
  op->caseType.accept(this);
  op->scanner.accept(this);
  op->contents.accept(this);
}

void IRVisitor::visit(const MemLoad* op) {
  op->lhsMem.accept(this);
  op->rhsMem.accept(this);
  op->start.accept(this);
  op->offset.accept(this);
  op->par.accept(this);
}


void IRVisitor::visit(const MemStore* op) {
  op->lhsMem.accept(this);
  op->rhsMem.accept(this);
  op->start.accept(this);
  op->offset.accept(this);
  op->par.accept(this);
}

void IRVisitor::visit(const GenBitVector* op) {
  op->shift.accept(this);
  op->out_bitcnt.accept(this);
  op->in_len.accept(this);
  op->in_fifo.accept(this);
  op->out_fifo.accept(this);
}

void IRVisitor::visit(const Scan* op) {
  op->par.accept(this);
  op->bitcnt.accept(this);
  op->in_fifo2.accept(this);
  if (op->in_fifo2.defined())
    op->in_fifo2.accept(this);
}

void IRVisitor::visit(const TypeCase* op) {
  for (auto v:op->vars)
    v.accept(this);
}

void IRVisitor::visit(const RMW* op) {
  op->arr.accept(this);
  op->addr.accept(this);
  op->data.accept(this);
  if (op->barrier.defined())
    op->barrier.accept(this);
}

void IRVisitor::visit(const FuncEnv* op) {
  op->env.accept(this);
}

void IRVisitor::visit(const AccelEnv* op) {
  op->aenv.accept(this);
}

/// SPATIAL ONLY END

}  // namespace ir
}  // namespace taco
