#include "taco/index_notation/index_notation_visitor.h"
#include "taco/index_notation/index_notation_nodes.h"

namespace taco {

// class ExprVisitorStrict
IndexExprVisitorStrict::~IndexExprVisitorStrict() {
}

void IndexExprVisitorStrict::visit(const IndexExpr& expr) {
  expr.accept(this);
}

// class IndexNotationVisitorStrict
IndexNotationVisitorStrict::~IndexNotationVisitorStrict() {
}

void IndexNotationVisitorStrict::visit(const IndexStmt& expr) {
  expr.accept(this);
}


// class ExprVisitor
IndexNotationVisitor::~IndexNotationVisitor() {
}

void IndexNotationVisitor::visit(const AccessNode* op) {
}

void IndexNotationVisitor::visit(const NegNode* op) {
  visit(static_cast<const UnaryExprNode*>(op));
}

void IndexNotationVisitor::visit(const SqrtNode* op) {
  visit(static_cast<const UnaryExprNode*>(op));
}

void IndexNotationVisitor::visit(const AddNode* op) {
  visit(static_cast<const BinaryExprNode*>(op));
}

void IndexNotationVisitor::visit(const SubNode* op) {
  visit(static_cast<const BinaryExprNode*>(op));
}

void IndexNotationVisitor::visit(const MulNode* op) {
  visit(static_cast<const BinaryExprNode*>(op));
}

void IndexNotationVisitor::visit(const DivNode* op) {
  visit(static_cast<const BinaryExprNode*>(op));
}

void IndexNotationVisitor::visit(const IntImmNode* op) {
  visit(static_cast<const ImmExprNode*>(op));
}

void IndexNotationVisitor::visit(const FloatImmNode* op) {
  visit(static_cast<const ImmExprNode*>(op));
}

void IndexNotationVisitor::visit(const ComplexImmNode* op) {
  visit(static_cast<const ImmExprNode*>(op));
}

void IndexNotationVisitor::visit(const UIntImmNode* op) {
  visit(static_cast<const ImmExprNode*>(op));
}

void IndexNotationVisitor::visit(const ImmExprNode*) {
}

void IndexNotationVisitor::visit(const UnaryExprNode* op) {
  op->a.accept(this);
}

void IndexNotationVisitor::visit(const BinaryExprNode* op) {
  op->a.accept(this);
  op->b.accept(this);
}

void IndexNotationVisitor::visit(const ReductionNode* op) {
  op->a.accept(this);
}

void IndexNotationVisitor::visit(const AssignmentNode* op) {
  op->rhs.accept(this);
}

void IndexNotationVisitor::visit(const ForallNode* op) {
  op->stmt.accept(this);
}

void IndexNotationVisitor::visit(const WhereNode* op) {
  op->producer.accept(this);
  op->consumer.accept(this);
}

void IndexNotationVisitor::visit(const MultiNode* op) {
  op->stmt1.accept(this);
  op->stmt2.accept(this);
}

void IndexNotationVisitor::visit(const SequenceNode* op) {
  op->definition.accept(this);
  op->mutation.accept(this);
}

}
