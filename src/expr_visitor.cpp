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
  op->a.accept(this);
}

void ExprVisitor::visit(const Sqrt* op) {
  op->a.accept(this);
}

void ExprVisitor::visit(const Add* op) {
  op->a.accept(this);
  op->b.accept(this);
}

void ExprVisitor::visit(const Sub* op) {
  op->a.accept(this);
  op->b.accept(this);
}

void ExprVisitor::visit(const Mul* op) {
  op->a.accept(this);
  op->b.accept(this);
}

void ExprVisitor::visit(const Div* op) {
  op->a.accept(this);
  op->b.accept(this);
}

}}
