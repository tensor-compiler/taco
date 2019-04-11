#ifndef TACO_LINEAR_H
#define TACO_LINEAR_H

#include "taco/tensor.h"

namespace taco {
namespace linear {

struct LinearIndexExpr {
  virtual void setFirstIndexVar(IndexVar first)=0;
  virtual void setSecondIndexVar(IndexVar second)=0;
};

struct MatrixIndexExpr : public LinearIndexExpr {
  void setFirstIndexVar(IndexVar first) {
    i = first;
  }
  void setSecondIndexVar(IndexVar second) {
    j = second;
  }

  IndexVar getFirstIndexVar() {
    return i;
  }

  IndexVar getSecondIndexVar() {
    return j;
  }

  IndexVar i;
  IndexVar j;
};

struct VectorIndexExpr : public LinearIndexExpr {
  void setFirstIndexVar(IndexVar first) {
    i = first;
  }
  void setSecondIndexVar(IndexVar second) {
    return;
  }

  IndexVar getFirstIndexVar() {
    return i;
  }

  IndexVar i;
};

struct ScalarIndexExpr : public LinearIndexExpr {
  void setFirstIndexVar(IndexVar first) {
  	return;
  }
  void setSecondIndexVar(IndexVar second) {
    return;
  }
};

/// Informs the IndexExpression about the operands for visit resolution.
struct MatrixPlusMatrix : public AddNode, public MatrixIndexExpr {
  AddMatrixPlusMatrix(IndexExpr a, IndexExpr b) : AddNode(a, b) {}
};

/// Informs the IndexExpression about the operands for visit resolution.
struct MatrixTimesMatrix : public MulNode, public MatrixIndexExpr {
  MulMatrixPlusMatrix(IndexExpr a, IndexExpr b) : MulNode(a, b) {}
};

/// Informs the IndexExpression about the operands for visit resolution.
struct VectorPlusVector : public AddNode, public VectorIndexExpr {
  AddMatrixPlusMatrix(IndexExpr a, IndexExpr b) : AddNode(a, b) {}
};

/// Informs the IndexExpression about the operands for visit resolution.
struct MatrixTimesVector : public MulNode, public VectorIndexExpr {
  MulMatrixPlusMatrix(IndexExpr a, IndexExpr b) : MulNode(a, b) {}
};

/// Informs the IndexExpression about the operands for visit resolution.
struct ScalarPlusScalar : public AddNode, public ScalarIndexExpr {
  AddMatrixPlusMatrix(IndexExpr a, IndexExpr b) : AddNode(a, b) {}
};

/// Informs the IndexExpression about the operands for visit resolution.
struct ScalarTimesScalar : public MulNode, public ScalarIndexExpr {
  MulMatrixPlusMatrix(IndexExpr a, IndexExpr b) : MulNode(a, b) {}
};

/// Informs the IndexExpression about the operands for visit resolution.
struct ScalarTimesVector : public MulNode, public VectorIndexExpr {
  MulMatrixPlusMatrix(IndexExpr a, IndexExpr b) : MulNode(a, b) {}
};

/// Informs the IndexExpression about the operands for visit resolution.
struct ScalarTimesMatrix : public MulNode, public MatrixIndexExpr {
  MulMatrixPlusMatrix(IndexExpr a, IndexExpr b) : MulNode(a, b) {}
};

// TODO remove this when removing the old dense
IndexExpr setIndexVars(const IndexExpr& expr) {
  struct SetIndexVars : IndexNotationRewriter {
    using IndexNotationRewriter::visit;
    void visit(const AddNode* node) {
      if (isa<MatrixPlusMatrix>(node)) {
        IndexVar i = to<MatrixPlusMatrix>(node)->getFirstIndexVar();
        IndexVar j = to<MatrixPlusMatrix>(node)->getSecondIndexVar();
        IndexExpr A = node.getA();
        to<MatrixIndexExpr>(A)->setFirstIndexVar(i);
        to<MatrixIndexExpr>(A)->setSecondIndexVar(j);
        A = rewrite(A);
        IndexExpr B = node.getB();
        to<MatrixIndexExpr>(B)->setFirstIndexVar(i);
        to<MatrixIndexExpr>(B)->setSecondIndexVar(j);
        B = rewrite(B);
        expr = Add(A,B);
      } else if (isa<VectorPlusVector>(node)) {
      	IndexVar i = to<VectorPlusVector>(node)->getFirstIndexVar();
        IndexExpr a = node.getA();
        to<VectorIndexExpr>(a)->setFirstIndexVar(i);
        a = rewrite(a);
        IndexExpr b = node.getB();
        to<VectorIndexExpr>(b)->setFirstIndexVar(i);
        b = rewrite(b);
        expr = Add(a,b);
      } else {  // isa<ScalarPlusScalar>(node)
      	expr = node;
      }
    }

    void visit(const MulNode* node) {
      if (isa<MatrixTimesMatrix>(node)) {
        IndexVar i = to<MatrixTimesMatrix>(node)->getFirstIndexVar();
        IndexVar j = to<MatrixTimesMatrix>(node)->getSecondIndexVar();
        IndexVar k;
        IndexExpr A = node.getA();
        to<MatrixIndexExpr>(A)->setFirstIndexVar(i);
        to<MatrixIndexExpr>(A)->setSecondIndexVar(k);
        A = rewrite(A);
        IndexExpr B = node.getB();
        to<MatrixIndexExpr>(B)->setFirstIndexVar(k);
        to<MatrixIndexExpr>(B)->setSecondIndexVar(j);
        B = rewrite(B);
        expr = Mul(A,B);
      } else if (isa<MatrixTimesVector>(node)) {
      	IndexVar i = to<MatrixTimesVector>(node)->getFirstIndexVar();
      	IndexVar j;
        IndexExpr A = node.getA();
        to<MatrixIndexExpr>(A)->setFirstIndexVar(i);
        to<MatrixIndexExpr>(A)->setSecondIndexVar(j);
        A = rewrite(A);
        IndexExpr b = node.getB();
        to<VectorIndexExpr>(b)->setFirstIndexVar(j);
        b = rewrite(b);
        expr = Mul(A,b);
      } else if (isa<ScalarTimesMatrix>(node)) {
      	IndexVar i = to<ScalarTimesMatrix>(node)->getFirstIndexVar();
      	IndexVar j = to<ScalarTimesMatrix>(node)->getSecondIndexVar();
        IndexExpr a = node.getA();
        IndexExpr B = node.getB();
        to<MatrixIndexExpr>(B)->setFirstIndexVar(i);
        to<MatrixIndexExpr>(B)->setSecondIndexVar(j);
        B = rewrite(B);
        expr = Mul(a,B);
      } else if (isa<ScalarTimesVector>(node)) {
      	IndexVar i = to<ScalarTimesVector>(node)->getFirstIndexVar();
        IndexExpr a = node.getA();
        IndexExpr b = node.getB();
        to<MatrixIndexExpr>(b)->setFirstIndexVar(i);
        b = rewrite(b);
        expr = Mul(a,b);
      } else {  // isa<ScalarTimesScalar>(node)
      	expr = node;
      }
    }
  };
  return SetIndexVars().rewrite(expr);
}

IndexExpr setIndexVars(const IndexExpr& expr) {
  struct SetIndexVars : public IndexNotationVisitor {
    using IndexNotationVisitor::visit;
    set<TensorBase> inserted;
    vector<TensorBase> operands;
    void visit(const AccessNode* node) {
      taco_iassert(isa<AccessTensorNode>(node)) << "Unknown subexpression";
      TensorBase tensor = to<AccessTensorNode>(node)->tensor;
      if (!util::contains(inserted, tensor)) {
        inserted.insert(tensor);
        operands.push_back(tensor);
      }
    }
  };
  GetOperands getOperands;
  expr.accept(&getOperands);
  return getOperands.operands;
}

struct MatrixExpr {
  MatrixExpr(TensorBase * tensor) {
    IndexVar i, j;
    expr = tensor->operator()(i,j);
  }
  IndexExpr expr;
};

struct VectorExpr {
  VectorExpr(TensorBase * tensor) {
    IndexVar i;
    expr = tensor->operator()(i);
  }
  IndexExpr expr;
};

struct ScalarExpr {
  ScalarExpr(TensorBase * tensor) {
    expr = tensor->operator()();
  }
  IndexExpr expr;
};

MatrixExpr operator*(const MatrixExpr& a, const MatrixExpr& b);
MatrixExpr operator+(const MatrixExpr& a, const MatrixExpr& b);

VectorExpr operator+(const VectorExpr& a, const VectorExpr& b);

ScalarExpr operator*(const ScalarExpr& a, const ScalarExpr& b);
ScalarExpr operator+(const ScalarExpr& a, const ScalarExpr& b);

VectorExpr operator*(const ScalarVar& a, const VectorVar& b);
MatrixExpr operator*(const ScalarVar& a, const MatrixVar& b);
VectorExpr operator*(const MatrixVar& a, const VectorVar& b);


template <typename CType>
class Matrix : public Tensor<CType> {
public:
  /// Create a tensor with the given dimensions and format
  Matrix(std::vector<int> dimensions, Format format)
      : Tensor<CType>(dimensions, format) {
    taco_uassert(2 == (size_t)Tensor<CType>::getOrder())
        << "Matrices must be created with an order 2 format, "
        << "but instead format of order "
        << Tensor<CType>::getOrder() << " was used.";
  }

  operator MatrixVar() {
    return MatrixVar(this);
  }

  /// Create an index expression that accesses (reads) this tensor.
  const Access operator()(const std::vector<IndexVar>& indices) const {
    return Tensor<CType>::operator()(indices);
  }

  /// Create an index expression that accesses (reads or writes) this tensor.
  Access operator()(const std::vector<IndexVar>& indices) {
    return Tensor<CType>::operator()(indices);
  }

  const Access operator()(const IndexVar& i, const IndexVar& j) const {
    return Tensor<CType>::operator()(i, j);
  }

  /// Create an index expression that accesses (reads) this tensor.
  Access operator()(const IndexVar& i, const IndexVar& j) {
    return Tensor<CType>::operator()(i, j);
  }

  ScalarAccess<CType> operator()(int i, int j) {
    return Tensor<CType>::operator()(i, j);
  }

private:
  using Tensor<CType>::operator();
};

template <typename CType>
class Vector : public Tensor<CType> {
public:
  /// Create a tensor with the given dimensions and format
  Vector(std::vector<int> dimensions, Format format)
      : Tensor<CType>(dimensions, format) {
    taco_uassert(1 == (size_t)Tensor<CType>::getOrder())
        << "Vectors must be created with an order 1 format, "
        << "but instead format of order "
        << Tensor<CType>::getOrder() << " was used.";
  }

  operator VectorVar() {
    return VectorVar(this);
  }

  /// Create an index expression that accesses (reads) this tensor.
  const Access operator()(const std::vector<IndexVar>& indices) const {
    return Tensor<CType>::operator()(indices);
  }

  /// Create an index expression that accesses (reads or writes) this tensor.
  Access operator()(const std::vector<IndexVar>& indices) {
    return Tensor<CType>::operator()(indices);
  }

  const Access operator()(const IndexVar& i) const {
    return Tensor<CType>::operator()(i);
  }

  /// Create an index expression that accesses (reads) this tensor.
  Access operator()(const IndexVar& i) {
    return Tensor<CType>::operator()(i);
  }

  ScalarAccess<CType> operator()(int i) {
    return Tensor<CType>::operator()(i);
  }

private:
  using Tensor<CType>::operator();
};

template <typename CType>
class Scalar : public Tensor<CType> {
public:
  /// Create a tensor with the given dimensions and format
  Scalar() : Tensor<CType>() {}

  operator ScalarVar() {
    return ScalarVar(this);
  }

  /// Create an index expression that accesses (reads) this tensor.
  const Access operator()() const {
    return Tensor<CType>::operator()();
  }

private:
  using Tensor<CType>::operator();
};


}
}
#endif