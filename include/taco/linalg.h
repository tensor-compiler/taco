#ifndef TACO_LINALG_H
#define TACO_LINALG_H

#include "taco/type.h"
#include "taco/tensor.h"
#include "taco/format.h"

#include "taco/linalg_notation/linalg_notation.h"

namespace taco {

class LinalgBase : public LinalgExpr {
  std::string name;
  Datatype ctype;
public:
  LinalgBase(std::string name, Datatype ctype);

  /* LinalgBase operator=(LinalgExpr) { */
  /*   return (LinalgBase)LinalgExpr; */
  /* } */

};

// ------------------------------------------------------------
// Matrix class
// ------------------------------------------------------------

template <typename CType>
class Matrix : public LinalgBase {
  public:
    explicit Matrix(std::string name);
};

// ------------------------------------------------------------
// Matrix template method implementations
// ------------------------------------------------------------

template <typename CType>
Matrix<CType>::Matrix(std::string name) : LinalgBase(name, type<CType>()) {}

// ------------------------------------------------------------
// Vector class
// ------------------------------------------------------------

template <typename CType>
class Vector : public LinalgBase {
  public:
    explicit Vector(std::string name);
};

// ------------------------------------------------------------
// Vector template method implementations
// ------------------------------------------------------------

template <typename CType>
Vector<CType>::Vector(std::string name) : LinalgBase(name, type<CType>()) {}
}
#endif
