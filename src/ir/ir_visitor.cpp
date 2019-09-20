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
  op->a.accept(this);
  op->b.accept(this);
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

void IRVisitor::visit(const IfThenElse* op) {
  op->cond.accept(this);
  op->then.accept(this);

  if (op->otherwise.defined()) {
    op->otherwise.accept(this);
  }
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

void IRVisitor::visit(const Print* op) {
  for (auto e: op->params)
    e.accept(this);
}

}  // namespace ir
}  // namespace taco
