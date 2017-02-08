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

void ExprVisitor::visit(const Read* op) {
}

void ExprVisitor::visit(const Neg* op) {
  visit(static_cast<const UnaryExpr*>(op));
}

void ExprVisitor::visit(const Sqrt* op) {
  visit(static_cast<const UnaryExpr*>(op));
}

void ExprVisitor::visit(const Add* op) {
  visit(static_cast<const BinaryExpr*>(op));
}

void ExprVisitor::visit(const Sub* op) {
  visit(static_cast<const BinaryExpr*>(op));
}

void ExprVisitor::visit(const Mul* op) {
  visit(static_cast<const BinaryExpr*>(op));
}

void ExprVisitor::visit(const Div* op) {
  visit(static_cast<const BinaryExpr*>(op));
}

void ExprVisitor::visit(const IntImm*) {
}

void ExprVisitor::visit(const FloatImm*) {
}

void ExprVisitor::visit(const DoubleImm*) {
}

void ExprVisitor::visit(const ImmExpr*) {
}

void ExprVisitor::visit(const UnaryExpr* op) {
  op->a.accept(this);
}

void ExprVisitor::visit(const BinaryExpr* op) {
  op->a.accept(this);
  op->b.accept(this);
}

}}
