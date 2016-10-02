#include "ir_visitor.h"

namespace tacit {
namespace internal {

IRVisitor::~IRVisitor() {
}

void IRVisitor::visit(const Literal*) {
}

void IRVisitor::visit(const Var*) {
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
  op->a.accept(this);
  op->b.accept(this);
}

void IRVisitor::visit(const Max* op){
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

void IRVisitor::visit(const IfThenElse* op) {
  op->cond.accept(this);
  op->then.accept(this);
  op->otherwise.accept(this);
}

void IRVisitor::visit(const Load* op) {
  op->arr.accept(this);
  op->loc.accept(this);
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

void IRVisitor::visit(const Block* op) {
  for (auto s:op->contents)
    s.accept(this);
}

void IRVisitor::visit(const Function* op) {
  op->body.accept(this);
}

}  // namespace tacit
}  // namespace internal
