#include "expr_visitor.h"

#include "expr_nodes.h"

namespace taco {
namespace internal {

// class ExprVisitorStrict
ExprVisitorStrict::~ExprVisitorStrict() {
}

// class ExprVisitor
ExprVisitor::~ExprVisitor() {
}

void ExprVisitor::visit(const IntImm*) {
}

void ExprVisitor::visit(const FloatImm*) {
}

void ExprVisitor::visit(const DoubleImm*) {
}

void ExprVisitor::visit(const Read* op) {
}

void ExprVisitor::visit(const Neg* op) {
  op->operand.accept(this);
}

void ExprVisitor::visit(const Sqrt* op) {
  op->operand.accept(this);
}

void ExprVisitor::visit(const Add* op) {
  op->lhs.accept(this);
  op->rhs.accept(this);
}

void ExprVisitor::visit(const Sub* op) {
  op->lhs.accept(this);
  op->rhs.accept(this);
}

void ExprVisitor::visit(const Mul* op) {
  op->lhs.accept(this);
  op->rhs.accept(this);
}

void ExprVisitor::visit(const Div* op) {
  op->lhs.accept(this);
  op->rhs.accept(this);
}

}}
