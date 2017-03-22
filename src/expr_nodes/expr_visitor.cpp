#include "taco/expr_nodes/expr_visitor.h"

#include "taco/expr_nodes/expr_nodes.h"

namespace taco {
namespace expr_nodes {

// class ExprVisitorStrict
ExprVisitorStrict::~ExprVisitorStrict() {
}

// class ExprVisitor
ExprVisitor::~ExprVisitor() {
}

void ExprVisitor::visit(const ReadNode* op) {
}

void ExprVisitor::visit(const NegNode* op) {
  visit(static_cast<const UnaryExprNode*>(op));
}

void ExprVisitor::visit(const SqrtNode* op) {
  visit(static_cast<const UnaryExprNode*>(op));
}

void ExprVisitor::visit(const AddNode* op) {
  visit(static_cast<const BinaryExprNode*>(op));
}

void ExprVisitor::visit(const SubNode* op) {
  visit(static_cast<const BinaryExprNode*>(op));
}

void ExprVisitor::visit(const MulNode* op) {
  visit(static_cast<const BinaryExprNode*>(op));
}

void ExprVisitor::visit(const DivNode* op) {
  visit(static_cast<const BinaryExprNode*>(op));
}

void ExprVisitor::visit(const IntImmNode*) {
}

void ExprVisitor::visit(const FloatImmNode*) {
}

void ExprVisitor::visit(const DoubleImmNode*) {
}

void ExprVisitor::visit(const ImmExprNode*) {
}

void ExprVisitor::visit(const UnaryExprNode* op) {
  op->a.accept(this);
}

void ExprVisitor::visit(const BinaryExprNode* op) {
  op->a.accept(this);
  op->b.accept(this);
}

}}
