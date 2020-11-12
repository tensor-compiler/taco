#include "taco/index_notation/index_notation.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>
#include <utility>
#include <set>
#include <taco/ir/simplify.h>
#include "lower/mode_access.h"

#include "error/error_checks.h"
#include "taco/error/error_messages.h"
#include "taco/type.h"
#include "taco/format.h"

#include "taco/index_notation/intrinsic.h"
#include "taco/index_notation/schedule.h"
#include "taco/index_notation/transformations.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/linalg_notation/linalg_notation_nodes.h"
#include "taco/index_notation/index_notation_rewriter.h"
#include "taco/linalg_notation/linalg_notation_printer.h"
#include "taco/ir/ir.h"
#include "taco/lower/lower.h"
#include "taco/codegen/module.h"

#include "taco/util/name_generator.h"
#include "taco/util/scopedmap.h"
#include "taco/util/strings.h"
#include "taco/util/collections.h"

using namespace std;

namespace taco {

LinalgExpr::LinalgExpr(TensorVar var) : LinalgExpr(new LinalgVarNode(var)) {
}

LinalgExpr::LinalgExpr(TensorVar var, bool isColVec, TensorBase* _tensorBase) : LinalgExpr(new LinalgTensorBaseNode(var, _tensorBase)) {
  tensorBase = _tensorBase;
}

LinalgExpr::LinalgExpr(TensorVar var, bool isColVec) : LinalgExpr(new LinalgVarNode(var, isColVec)) {
}

LinalgExpr::LinalgExpr(char val) : LinalgExpr(new LinalgLiteralNode(val)) {
}

LinalgExpr::LinalgExpr(int8_t val) : LinalgExpr(new LinalgLiteralNode(val)) {
}

LinalgExpr::LinalgExpr(int16_t val) : LinalgExpr(new LinalgLiteralNode(val)) {
}

LinalgExpr::LinalgExpr(int32_t val) : LinalgExpr(new LinalgLiteralNode(val)) {
}

LinalgExpr::LinalgExpr(int64_t val) : LinalgExpr(new LinalgLiteralNode(val)) {
}

LinalgExpr::LinalgExpr(uint8_t val) : LinalgExpr(new LinalgLiteralNode(val)) {
}

LinalgExpr::LinalgExpr(uint16_t val) : LinalgExpr(new LinalgLiteralNode(val)) {
}

LinalgExpr::LinalgExpr(uint32_t val) : LinalgExpr(new LinalgLiteralNode(val)) {
}

LinalgExpr::LinalgExpr(uint64_t val) : LinalgExpr(new LinalgLiteralNode(val)) {
}

LinalgExpr::LinalgExpr(float val) : LinalgExpr(new LinalgLiteralNode(val)) {
}

LinalgExpr::LinalgExpr(double val) : LinalgExpr(new LinalgLiteralNode(val)) {
}

LinalgExpr::LinalgExpr(std::complex<float> val) : LinalgExpr(new LinalgLiteralNode(val)) {
}

LinalgExpr::LinalgExpr(std::complex<double> val) : LinalgExpr(new LinalgLiteralNode(val)) {
}

Datatype LinalgExpr::getDataType() const {
  return const_cast<LinalgExprNode*>(this->ptr)->getDataType();
}

int LinalgExpr::getOrder() const {
  return const_cast<LinalgExprNode*>(this->ptr)->getOrder();
}

bool LinalgExpr::isColVector() const {
  return const_cast<LinalgExprNode*>(this->ptr)->isColVector();
}

void LinalgExpr::accept(LinalgExprVisitorStrict *v) const {
  ptr->accept(v);
}

std::ostream& operator<<(std::ostream& os, const LinalgExpr& expr) {
  if (!expr.defined()) return os << "LinalgExpr()";
  LinalgNotationPrinter printer(os);
  printer.print(expr);
  return os;
}

void checkCompatibleShape(const LinalgExpr &lhs, const LinalgExpr &rhs) {
  taco_uassert(lhs.getOrder() == rhs.getOrder()) << "RHS and LHS order do not match for linear algebra "
                                                    "binary operation" << endl;
  if (lhs.getOrder() == 1)
    taco_uassert(lhs.isColVector() == rhs.isColVector()) << "RHS and LHS vector type do not match for linear algebra "
                                                            "binary operation" << endl;
}

LinalgExpr operator-(const LinalgExpr &expr) {
  return new LinalgNegNode(expr.ptr);
}

LinalgExpr operator+(const LinalgExpr &lhs, const LinalgExpr &rhs) {
  checkCompatibleShape(lhs, rhs);
  return new LinalgAddNode(lhs, rhs, lhs.getOrder(), lhs.isColVector());
}

LinalgExpr operator-(const LinalgExpr &lhs, const LinalgExpr &rhs) {
  checkCompatibleShape(lhs, rhs);
  return new LinalgSubNode(lhs, rhs, lhs.getOrder(), lhs.isColVector());
}

LinalgExpr operator*(const LinalgExpr &lhs, const LinalgExpr &rhs) {
  int order = 0;
  bool isColVec = false;
  // Matrix-matrix mult
  if (lhs.getOrder() == 2 && rhs.getOrder() == 2) {
    order = 2;
  }
  // Matrix-column vector multiply
  else if (lhs.getOrder() == 2 && rhs.getOrder() == 1 && rhs.isColVector()) {
    order = 1;
    isColVec = true;
  }
  // Row-vector Matrix multiply
  else if (lhs.getOrder() == 1 && !lhs.isColVector() && rhs.getOrder() == 2) {
    order = 1;
  }
  // Inner product
  else if (lhs.getOrder() == 1 && !lhs.isColVector() && rhs.getOrder() == 1 && rhs.isColVector()) {
    order = 0;
  }
  // Outer product
  else if (lhs.getOrder() == 1 && lhs.isColVector() && rhs.getOrder() == 1 && !rhs.isColVector()) {
    order = 2;
  }
  else {
    taco_uassert(lhs.getOrder() != rhs.getOrder()) << "RHS and LHS order/vector type do not match "
                                                      "for linear algebra matrix multiply" << endl;
  }
  return new LinalgMatMulNode(lhs, rhs, order, isColVec);
}

LinalgExpr operator/(const LinalgExpr &lhs, const LinalgExpr &rhs) {
  checkCompatibleShape(lhs, rhs);
  return new LinalgDivNode(lhs, rhs, lhs.getOrder(), lhs.isColVector());
}

LinalgExpr elemMul(const LinalgExpr &lhs, const LinalgExpr &rhs) {
  checkCompatibleShape(lhs, rhs);
  return new LinalgElemMulNode(lhs, rhs, lhs.getOrder(), lhs.isColVector());
}

LinalgExpr transpose(const LinalgExpr &lhs) {
  return new LinalgTransposeNode(lhs);
}

// class LinalgStmt
LinalgStmt::LinalgStmt() : util::IntrusivePtr<const LinalgStmtNode>(nullptr) {
}

LinalgStmt::LinalgStmt(const LinalgStmtNode* n)
  : util::IntrusivePtr<const LinalgStmtNode>(n) {
}

void LinalgStmt::accept(LinalgStmtVisitorStrict *v) const {
  ptr->accept(v);
}


// class LinalgAssignment
LinalgAssignment::LinalgAssignment(const LinalgAssignmentNode* n) : LinalgStmt(n) {
}

LinalgAssignment::LinalgAssignment(TensorVar lhs, LinalgExpr rhs)
  : LinalgAssignment(new LinalgAssignmentNode(lhs, rhs)) {
}

TensorVar LinalgAssignment::getLhs() const {
  return getNode(*this)->lhs;
}

LinalgExpr LinalgAssignment::getRhs() const {
  return getNode(*this)->rhs;
}

}   // namespace taco
