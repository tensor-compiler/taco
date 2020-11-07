#ifndef TACO_LINALG_H
#define TACO_LINALG_H

#include "taco/type.h"
#include "taco/tensor.h"
#include "taco/format.h"

#include "taco/linalg_notation/linalg_notation.h"
#include "taco/linalg_notation/linalg_notation_nodes.h"

namespace taco {

class LinalgBase : public LinalgExpr {
  std::string name;
  Type tensorType;

  // The associated tensor
  TensorBase tbase;

  LinalgAssignment assignment;

  typedef VarNode Node;
public:
  LinalgBase(std::string name, Type tensorType);
  LinalgBase(std::string name, Type tensorType, Format format);
  /// [LINALG NOTATION]
  LinalgAssignment operator=(const LinalgExpr& expr);

  void ping() {
    std::cout << "ping" << std::endl;
  }

};

// ------------------------------------------------------------
// Matrix class
// ------------------------------------------------------------

template <typename CType>
class Matrix : public LinalgBase {
public:
  explicit Matrix(std::string name);
  Matrix(std::string name, size_t dim1, size_t dim2);
  Matrix(std::string name, std::vector<size_t> dimensions);
  Matrix(std::string name, size_t dim1, size_t dim2, Format format);
  Matrix(std::string name, std::vector<size_t> dimensions, Format format);
  Matrix(std::string name, size_t dim1, size_t dim2, ModeFormat format1, ModeFormat format2);
  Matrix(std::string name, Type tensorType);
  Matrix(std::string name, Type tensorType, Format format);
  LinalgAssignment operator=(const LinalgExpr& expr) {
    return LinalgBase::operator=(expr);
  }

  // Support some Read methods
  CType at(const size_t coord_x, const size_t coord_y);

};

// ------------------------------------------------------------
// Matrix template method implementations
// ------------------------------------------------------------

template <typename CType>
Matrix<CType>::Matrix(std::string name) : LinalgBase(name, Type(type<CType>(), {42, 42})) {}
template <typename CType>
Matrix<CType>::Matrix(std::string name, std::vector<size_t> dimensions) : LinalgBase(name, Type(type<CType>(), dimensions)) {}
template <typename CType>
Matrix<CType>::Matrix(std::string name, size_t dim1, size_t dim2) : LinalgBase(name, Type(type<CType>(), {dim1, dim2})) {}
template <typename CType>
Matrix<CType>::Matrix(std::string name, size_t dim1, size_t dim2, Format format) :
  LinalgBase(name, Type(type<CType>(), {dim1, dim2}), format) {}
template <typename CType>
Matrix<CType>::Matrix(std::string name, std::vector<size_t> dimensions, Format format) :
  LinalgBase(name, Type(type<CType>(), dimensions), format) {}
template <typename CType>
Matrix<CType>::Matrix(std::string name, size_t dim1, size_t dim2, ModeFormat format1, ModeFormat format2) :
  LinalgBase(name, Type(type<CType>(), {dim1, dim2}), Format({format1, format2})) {}
template <typename CType>
Matrix<CType>::Matrix(std::string name, Type tensorType) : LinalgBase(name, tensorType) {}
template <typename CType>
Matrix<CType>::Matrix(std::string name, Type tensorType, Format format) : LinalgBase(name, tensorType, format) {}

// Definition of Read methods
template <typename CType>
CType Matrix<CType>::at(const size_t coord_x, const size_t coord_y) {
  return 0;
}
// ------------------------------------------------------------
// Vector class
// ------------------------------------------------------------

template <typename CType>
class Vector : public LinalgBase {
  std::string name;
  Datatype ctype;
public:
  explicit Vector(std::string name);
  Vector(std::string name, size_t dim);
  Vector(std::string name, size_t dim, Format format);
  Vector(std::string name, size_t dim, ModeFormat format);
  Vector(std::string name, Type type, Format format);
  Vector(std::string name, Type type, ModeFormat format);
  LinalgAssignment operator=(const LinalgExpr& expr) {
    return LinalgBase::operator=(expr);
  }
};

// ------------------------------------------------------------
// Vector template method implementations
// ------------------------------------------------------------

template <typename CType>
Vector<CType>::Vector(std::string name) : LinalgBase(name, Type(type<CType>(), {42})) {}
template <typename CType>
Vector<CType>::Vector(std::string name, size_t dim) : LinalgBase(name, Type(type<CType>(), {dim})) {}
template <typename CType>
Vector<CType>::Vector(std::string name, size_t dim, Format format) : LinalgBase(name, Type(type<CType>(), {dim}), format) {}
template <typename CType>
Vector<CType>::Vector(std::string name, size_t dim, ModeFormat format) :
  LinalgBase(name, Type(type<CType>(), {dim}), Format(format)) {}
template <typename CType>
Vector<CType>::Vector(std::string name, Type type, Format format) :
  LinalgBase(name, type, format) {}
template <typename CType>
Vector<CType>::Vector(std::string name, Type type, ModeFormat format) :
  LinalgBase(name, type, Format(format)) {}
}
#endif
