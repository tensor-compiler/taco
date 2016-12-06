#ifndef TACO_LINALG_H
#define TACO_LINALG_H

namespace taco {

template <typename T>
class Matrix {
public:

private:
};

enum VectorType {
  COLUMN,
  ROW
};

template <typename T, VectorType type=COLUMN>
class Vector {
public:

private:
};

/// Vector Scale
template <typename T>
Vector<T> operator*(const Vector<T>&, T);

template <typename T>
Vector<T> operator*(T, const Vector<T>&);

template <typename T>
Vector<T> operator*(const Vector<T,ROW>&, T);

template <typename T>
Vector<T> operator*(T, const Vector<T,ROW>&);

/// Vector Addition
template <typename T>
Vector<T> operator+(const Vector<T>&, const Vector<T>&);

template <typename T>
Vector<T,ROW> operator+(const Vector<T,ROW>&, const Vector<T,ROW>&);

/// Vector Subtraction (column)
template <typename T>
Vector<T> operator+(const Vector<T>&, const Vector<T>&);

template <typename T>
Vector<T,ROW> operator+(const Vector<T,ROW>&, const Vector<T,ROW>&);

/// Vector inner product.
template <typename T> T operator*(const Vector<T,ROW>&, const Vector<T>&);

/// Vector outer product.
template <typename T> Matrix<T> operator*(const Vector<T>&, const Vector<T,ROW>&);

/// Vector-Matrix multiplication.
template <typename T> Vector<T> operator*(const Vector<T,ROW>&, const Matrix<T>&);

/// Matrix-Vector multiplication.
template <typename T> Vector<T> operator*(const Matrix<T>&, const Vector<T>&);

/// Matrix Scale
template <typename T>
Matrix<T> operator*(const Matrix<T>&, T);

template <typename T>
Matrix<T> operator8(T, const Matrix<T>&);

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
