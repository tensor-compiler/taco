#ifndef TACO_LINALG_H
#define TACO_LINALG_H

#include <string>

#include "tensor.h"

namespace taco {

enum VectorType {
  COLUMN,
  ROW
};

template <typename C, VectorType T=COLUMN>
class Vector : public Tensor<C> {
public:
  Vector(std::string name, std::vector<int> dimensions, Format format);
  Vector(std::vector<int> dimensions, Format format);

  Vector<C,T> operator*=(C) {
    not_supported_yet;
  }

  Vector<C,T> operator+=(const Vector<C,T>&) {
    not_supported_yet;
  }

  Vector<C,T> operator-=(const Vector<C,T>&) {
    not_supported_yet;
  }
};

template <typename C>
class Matrix : public Tensor<C> {
public:
  Matrix(std::string name, std::vector<int> dimensions, Format format);
  Matrix(std::vector<int> dimensions, Format format);

  Matrix<C> operator*=(C) {
    not_supported_yet;
  }

  Matrix<C> operator+=(const Matrix<C>&)  {
    not_supported_yet;
  }

  Matrix<C> operator-=(const Matrix<C>&) {
    not_supported_yet;
  }
};

/// Vector Negation
template <typename C, VectorType T>
Vector<C,T> operator-(const Vector<C,T>&)  {
  not_supported_yet;
}

/// Vector Scale
template <typename C, VectorType T>
Vector<C,T> operator*(const Vector<C,T>&, C) {
  not_supported_yet;
}

template <typename C, VectorType T>
Vector<C,T> operator*(C, const Vector<C,T>&) {
  not_supported_yet;
}

/// Vector Addition
template <typename C, VectorType T>
Vector<C,T> operator+(const Vector<C,T>&, const Vector<C,T>&) {
  not_supported_yet;
}

/// Vector Subtraction
template <typename C, VectorType T>
Vector<C,T> operator-(const Vector<C,T>&, const Vector<C,T>&) {
  not_supported_yet;
}

/// Vector inner product.
template <typename T>
T operator*(const Vector<T,ROW>&, const Vector<T>&) {
  not_supported_yet;
}

/// Vector outer product.
template <typename T>
Matrix<T> operator*(const Vector<T>&, const Vector<T,ROW>&) {
  not_supported_yet;
}


/// Matrix Negation
template <typename C>
Matrix<C> operator-(const Matrix<C>&) {
  not_supported_yet;
}

/// Matrix Scale
template <typename T>
Matrix<T> operator*(const Matrix<T>&, T) {
  not_supported_yet;
}

template <typename T>
Matrix<T> operator*(T, const Matrix<T>&) {
  not_supported_yet;
}

/// Vector-Matrix multiplication.
template <typename T>
Vector<T> operator*(const Vector<T,ROW>&, const Matrix<T>&) {
  not_supported_yet;
}

/// Matrix-Vector multiplication.
template <typename T>
Vector<T> operator*(const Matrix<T>&, const Vector<T>&) {
  not_supported_yet;
}

/// Matrix Addition
template <typename T>
Matrix<T> operator+(const Matrix<T>&, const Matrix<T>&) {
  not_supported_yet;
}

/// Matrix Subtraction
template <typename T>
Matrix<T> operator-(const Matrix<T>&, const Matrix<T>&) {
  not_supported_yet;
}

/// Matrix Multiplication.
template <typename T>
Matrix<T> operator*(const Matrix<T>&, const Matrix<T>&) {
  not_supported_yet;
}

}
#endif
