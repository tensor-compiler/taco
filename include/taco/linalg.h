#ifndef TACO_LINALG_H
#define TACO_LINALG_H

#include "taco/type.h"
#include "taco/tensor.h"
#include "taco/format.h"

#include "taco/linalg_notation/linalg_notation.h"
#include "taco/linalg_notation/linalg_notation_nodes.h"
#include "taco/linalg_notation/linalg_notation_printer.h"


namespace taco {

class LinalgBase : public LinalgExpr {
protected:
  std::string name;
  Type tensorType;

  // The associated tensor
  TensorBase *tbase;

  LinalgAssignment assignment;
  IndexStmt indexAssignment;

  int idxcount;
  int block;

  IndexExpr rewrite(LinalgExpr linalg, std::vector<IndexVar> indices);

  IndexVar getUniqueIndex();

  std::vector<IndexVar> getUniqueIndices(size_t order);

public:
  LinalgBase(std::string name, Type tensorType, Datatype dtype, std::vector<int> dims, Format format, bool isColVec = false);
  LinalgBase(std::string name, Type tensorType, Datatype dtype, std::vector<int> dims, Format format, int block, bool isColVec = false);
  LinalgBase(std::string name, Type tensorType, bool isColVec = false);
  LinalgBase(std::string name, Type tensorType, Format format, bool isColVec = false);
  LinalgBase(TensorBase* tensor, bool isColVec = false);

  /// [LINALG NOTATION]
  LinalgAssignment operator=(const LinalgExpr &expr);

  const LinalgAssignment getAssignment() const;

  const IndexStmt getIndexAssignment() const;

  bool isBlocked() const;

  IndexStmt rewrite();

  typedef LinalgVarNode Node;
};

std::ostream &operator<<(std::ostream &os, const LinalgBase &linalg);

IndexExpr rewrite(LinalgExpr linalg, std::vector<IndexVar>);

IndexStmt rewrite(LinalgStmt linalg);

// ------------------------------------------------------------
// Matrix class
// ------------------------------------------------------------

template<typename CType>
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

  Matrix(std::string name, size_t dim1, size_t dim2, ModeFormat format1, ModeFormat format2, int block_size);

  LinalgAssignment operator=(const LinalgExpr &expr) {
    return LinalgBase::operator=(expr);
  }

  bool isBlocked() {
    return LinalgBase::isBlocked();
  }
  // Support some Read methods
  CType at(int coord_x, int coord_y);

  // And a Write method
  void insert(int coord_x, int coord_y, CType value);
  void insert(int coord_x, int coord_y, int coord_bx, int coord_by, CType value);


};

// ------------------------------------------------------------
// Matrix template method implementations
// ------------------------------------------------------------

template<typename CType>
Matrix<CType>::Matrix(std::string name) : LinalgBase(name, Type(type<CType>(), {42, 42})) {}

template<typename CType>
Matrix<CType>::Matrix(std::string name, std::vector<size_t> dimensions) : LinalgBase(name, Type(type<CType>(), dimensions)) {}

template<typename CType>
Matrix<CType>::Matrix(std::string name, size_t dim1, size_t dim2) : LinalgBase(name, Type(type<CType>(), {dim1, dim2})) {}

template<typename CType>
Matrix<CType>::Matrix(std::string name, size_t dim1, size_t dim2, Format format) :
  LinalgBase(name, Type(type<CType>(), {dim1, dim2}), format) {}

template<typename CType>
Matrix<CType>::Matrix(std::string name, std::vector<size_t> dimensions, Format format) :
  LinalgBase(name, Type(type<CType>(), dimensions), format) {}

template<typename CType>
Matrix<CType>::Matrix(std::string name, size_t dim1, size_t dim2, ModeFormat format1, ModeFormat format2) :
  LinalgBase(name, Type(type<CType>(), {dim1, dim2}), type<CType>(), {(int)dim1, (int)dim2}, Format({format1, format2}), false) {}

template<typename CType>
Matrix<CType>::Matrix(std::string name, Type tensorType) : LinalgBase(name, tensorType) {}

template<typename CType>
Matrix<CType>::Matrix(std::string name, Type tensorType, Format format) : LinalgBase(name, tensorType, format) {}

template<typename CType>
Matrix<CType>::Matrix(std::string name, size_t dim1, size_t dim2, ModeFormat format1, ModeFormat format2, int block_size) :
  LinalgBase(name, Type(type<CType>(), {dim1/block_size, dim2/block_size, block_size, block_size}), type<CType>(),
    {(int)dim1/block_size, (int)dim2/block_size, block_size, block_size}, Format({format1, format2, dense, dense}), block_size, false) {
    taco_uassert(block_size >= 0) << "Block size must be non-negative" << std::endl;
    taco_uassert(dim1 % block_size == 0 && dim2 % block_size == 0) << "Dimensions must be a multiple of block size" << std::endl;
  }

// Definition of Read methods
template <typename CType>
CType Matrix<CType>::at(int coord_x, int coord_y) {
  return tensorBase->at<CType>({coord_x, coord_y});
}

// Definition of Write methods
template <typename CType>
void Matrix<CType>::insert(int coord_x, int coord_y, CType value) {
  std::cout << "Blocked " << isBlocked() << std::endl;
  std::cout << this->block << std::endl;
  if (isBlocked()) {
    std::cout << "blocked matrix" << std::endl;
    for (int bx = 0; bx < block; bx++)
      for (int by = 0; by < block; by++)
        tensorBase->insert({coord_x, coord_y, bx, by}, value);
  } else {
    tensorBase->insert({coord_x, coord_y}, value);
  }
}

// Blocked Write Method
template <typename CType>
void Matrix<CType>::insert(int coord_x, int coord_y, int coord_bx, int coord_by, CType value) {
  std::cout << "blocked matrix" << std::endl;
  std::cout << "Blocked " << isBlocked() << std::endl;
  std::cout << this->block << std::endl;
  tensorBase->insert({coord_x, coord_y, coord_bx, coord_by}, value);
}
// ------------------------------------------------------------
// Vector class
// ------------------------------------------------------------

template<typename CType>
class Vector : public LinalgBase {
  std::string name;
  Datatype ctype;
public:
  explicit Vector(std::string name, bool isColVec = true);

  Vector(std::string name, size_t dim, bool isColVec = true);

  Vector(std::string name, size_t dim, Format format, bool isColVec = true);

  Vector(std::string name, size_t dim, ModeFormat format, bool isColVec = true);

  Vector(std::string name, Type type, Format format, bool isColVec = true);

  Vector(std::string name, Type type, ModeFormat format, bool isColVec = true);

  Vector(std::string name, Type type, ModeFormat format, int blocking, bool isColVec = true);

  LinalgAssignment operator=(const LinalgExpr &expr) {
    return LinalgBase::operator=(expr);
  }

  // Support some Write methods
  void insert(int coord, CType value);

  // Support some Read methods too
  CType at(int coord);

  // Support Read Method with Blocking
  CType at(int bcoord0, int bcoord1);
};

// ------------------------------------------------------------
// Vector template method implementations
// ------------------------------------------------------------

template<typename CType>
Vector<CType>::Vector(std::string name, bool isColVec) : LinalgBase(name, Type(type<CType>(), {42}), isColVec) {}

template<typename CType>
Vector<CType>::Vector(std::string name, size_t dim, bool isColVec) : LinalgBase(name, Type(type<CType>(), {dim}),
                                                                             isColVec) {}

template<typename CType>
Vector<CType>::Vector(std::string name, size_t dim, Format format, bool isColVec) : LinalgBase(name,
                                                                                            Type(type<CType>(), {dim}),
                                                                                            type<CType>(),
                                                                                            {(int) dim},
                                                                                            format, isColVec) {}

template<typename CType>
Vector<CType>::Vector(std::string name, size_t dim, ModeFormat format, bool isColVec) :
  LinalgBase(name, Type(type<CType>(), {dim}), type<CType>(), {(int)dim}, Format(format), isColVec) {}

template<typename CType>
Vector<CType>::Vector(std::string name, Type type, Format format, bool isColVec) :
  LinalgBase(name, type, format, isColVec) {}

template<typename CType>
Vector<CType>::Vector(std::string name, Type type, ModeFormat format, bool isColVec) :
  LinalgBase(name, type, Format(format), isColVec) {}


// Vector write methods
template<typename CType>
void Vector<CType>::insert(int coord, CType value) {
  tensorBase->insert({coord}, value);
}

template <typename CType>
CType Vector<CType>::at(int coord) {
  return tensorBase->at<CType>({coord});
}

template<typename CType>
class Scalar : public LinalgBase {
  std::string name;
  Datatype ctype;
public:
  explicit Scalar(std::string name);
  Scalar(std::string name, bool useTensorBase);

  LinalgAssignment operator=(const LinalgExpr &expr) {
    return LinalgBase::operator=(expr);
  }

  /* operator int() const { return */ 
};

template<typename CType>
Scalar<CType>::Scalar(std::string name) : LinalgBase(name, Type(type<CType>(), {})) {}
template<typename CType>
Scalar<CType>::Scalar(std::string name, bool useTensorBase) :
  LinalgBase(name, Type(type<CType>(), {}) , type<CType>(), {}, Format(), false) {}

}   // namespace taco
#endif
