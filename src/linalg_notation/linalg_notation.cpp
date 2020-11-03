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

LinalgExpr::LinalgExpr(TensorVar var) : LinalgExpr(new VarNode(var)) {
}

LinalgExpr::LinalgExpr(char val) : LinalgExpr(new LiteralNode(val)) {
}

LinalgExpr::LinalgExpr(int8_t val) : LinalgExpr(new LiteralNode(val)) {
}

LinalgExpr::LinalgExpr(int16_t val) : LinalgExpr(new LiteralNode(val)) {
}

LinalgExpr::LinalgExpr(int32_t val) : LinalgExpr(new LiteralNode(val)) {
}

LinalgExpr::LinalgExpr(int64_t val) : LinalgExpr(new LiteralNode(val)) {
}

LinalgExpr::LinalgExpr(uint8_t val) : LinalgExpr(new LiteralNode(val)) {
}

LinalgExpr::LinalgExpr(uint16_t val) : LinalgExpr(new LiteralNode(val)) {
}

LinalgExpr::LinalgExpr(uint32_t val) : LinalgExpr(new LiteralNode(val)) {
}

LinalgExpr::LinalgExpr(uint64_t val) : LinalgExpr(new LiteralNode(val)) {
}

LinalgExpr::LinalgExpr(float val) : LinalgExpr(new LiteralNode(val)) {
}

LinalgExpr::LinalgExpr(double val) : LinalgExpr(new LiteralNode(val)) {
}

LinalgExpr::LinalgExpr(std::complex<float> val) : LinalgExpr(new LiteralNode(val)) {
}

LinalgExpr::LinalgExpr(std::complex<double> val) : LinalgExpr(new LiteralNode(val)) {
}

Datatype LinalgExpr::getDataType() const {
  return const_cast<LinalgExprNode*>(this->ptr)->getDataType();
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

LinalgExpr operator-(const LinalgExpr &expr) {
  return new NegNode(expr.ptr);
}

LinalgExpr operator+(const LinalgExpr &lhs, const LinalgExpr &rhs) {
  return new AddNode(lhs, rhs);
}

LinalgExpr operator-(const LinalgExpr &lhs, const LinalgExpr &rhs) {
  return new SubNode(lhs, rhs);
}

LinalgExpr operator*(const LinalgExpr &lhs, const LinalgExpr &rhs) {
  return new MatMulNode(lhs, rhs);
}

LinalgExpr operator/(const LinalgExpr &lhs, const LinalgExpr &rhs) {
  return new DivNode(lhs, rhs);
}
}   // namespace taco