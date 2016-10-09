#include "expr_visitor.h"

#include "expr.h"
#include "operator.h"

namespace taco {
namespace internal {

ExprVisitor::~ExprVisitor() {
}

//void ExprVisitor::visit(const ImmNode* op) {
//}

void ExprVisitor::visit(const ReadNode* op) {
}

void ExprVisitor::visit(const AddNode* op) {
  for (auto& operand : op->operands) {
    operand.accept(this);
  }
}

void ExprVisitor::visit(const SubNode* op) {
  op->lhs.accept(this);
  op->rhs.accept(this);
}

void ExprVisitor::visit(const MulNode* op) {
  for (auto& operand : op->operands) {
    operand.accept(this);
  }
}

void ExprVisitor::visit(const DivNode* op) {
  op->lhs.accept(this);
  op->rhs.accept(this);
}

}}
