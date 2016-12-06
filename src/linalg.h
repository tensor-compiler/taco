#ifndef TACO_LINALG_H
#define TACO_LINALG_H

#include "tensor.h"

namespace taco {

enum VectorType {
  COLUMN,
  ROW
};

template <typename C, VectorType T=COLUMN>
class Vector : public Tensor<C> {
public:

private:
};

template <typename C>
class Matrix : public Tensor<C> {
public:

private:
};

/// Vector Negation
template <typename C, VectorType T>
Vector<C,T> operator-(const Vector<C,T>&);

/// Vector Scale
template <typename C, VectorType T>
Vector<C,T> operator*(const Vector<C,T>&, C);

template <typename C, VectorType T>
Vector<C,T> operator*(C, const Vector<C,T>&);

/// Vector Addition
template <typename C, VectorType T>
Vector<C,T> operator+(const Vector<C,T>&, const Vector<C,T>&);

/// Vector Subtraction
template <typename C, VectorType T>
Vector<C,T> operator+(const Vector<C,T>&, const Vector<C,T>&);

/// Vector inner product.
template <typename T> T operator*(const Vector<T,ROW>&, const Vector<T>&);

/// Vector outer product.
template <typename T> Matrix<T> operator*(const Vector<T>&, const Vector<T,ROW>&);


/// Matrix Negation
template <typename C>
Matrix<C> operator-(const Matrix<C>&);

/// Matrix Scale
template <typename T>
Matrix<T> operator*(const Matrix<T>&, T);

template <typename T>
Matrix<T> operator*(T, const Matrix<T>&);

/// Vector-Matrix multiplication.
template <typename T> Vector<T> operator*(const Vector<T,ROW>&, const Matrix<T>&);

/// Matrix-Vector multiplication.
template <typename T> Vector<T> operator*(const Matrix<T>&, const Vector<T>&);

/// Matrix Addition
template <typename T>
Matrix<T> operator+(const Matrix<T>&, const Matrix<T>&);

/// Matrix Subtraction
template <typename T>
Matrix<T> operator-(const Matrix<T>&, const Matrix<T>&);

/// Matrix Multiplication.
template <typename T> Matrix<T> operator*(const Matrix<T>&, const Matrix<T>&);

}
#endif
