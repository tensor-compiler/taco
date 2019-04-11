#include "taco/linear_algebra/linear_algebra.h"

namespace taco {
namespace linear {

// Matrix Matrix operations

MatrixExpr operator*(const MatrixExpr& a, const MatrixExpr& b) {
  MatrixExpr matrixExpr;
  matrixExpr.expr = MatrixTimesMatrix(a.expr, b.expr);
  return matrixExpr;
}

MatrixExpr operator+(const MatrixExpr& a, const MatrixExpr& b) {
  MatrixExpr matrixExpr;
  matrixExpr.expr = MatrixPlusMatrix(a.expr, b.expr);
  return matrixExpr;
}

// Vector Vector operations

VectorExpr operator+(const VectorExpr& a, const VectorExpr& b) {
  VectorExpr vectorExpr;
  vectorExpr.expr = VectorPlusVector(a.expr, b.expr);
  return vectorExpr;
}

// Scalar Scalar operations

ScalarExpr operator*(const ScalarExpr& a, const ScalarExpr& b) {
  ScalarExpr scalarExpr;
  scalarExpr.expr = ScalarTimesScalar(a.expr, b.expr);
  return scalarExpr;
}

ScalarExpr operator+(const ScalarExpr& a, const ScalarExpr& b) {
  ScalarExpr scalarExpr;
  scalarExpr.expr = ScalarPlusScalar(a.expr, b.expr);
  return scalarExpr;
}

// Mixed operations

MatrixExpr operator*(const ScalarExpr& a, const MatrixExpr& b) {
  MatrixExpr matrixExpr;
  matrixExpr.expr = ScalarTimesMatrix(a.expr, b.expr);
  return matrixExpr;
}

VectorExpr operator*(const ScalarExpr& a, const VectorExpr& b) {
  VectorExpr vectorExpr;
  vectorExpr.expr = ScalarTimesVector(a.expr, b.expr);
  return vectorExpr;
}

VectorExpr operator*(const MatrixExpr& a, const VectorExpr& b) {
  VectorExpr vectorExpr;
  vectorExpr.expr = MatrixTimesVector(a.expr, b.expr);
  return vectorExpr;
}

}
}