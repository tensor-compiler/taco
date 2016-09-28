#ifndef TAC_SCALAR_TYPES_H
#define TAC_SCALAR_TYPES_H

#include "error.h"

namespace tac {
namespace internal {

enum class ScalarType {Int, Float, Double};

template <typename T> inline ScalarType typeOf() {
  ierror << "Unsupported type";
  return ScalarType::Double;
}

template <> inline ScalarType typeOf<int>() {
  return ScalarType::Int;
}

template <> inline ScalarType typeOf<float>() {
  return ScalarType::Float;
}

template <> inline ScalarType typeOf<double>() {
  return ScalarType::Double;
}

}}

#endif
