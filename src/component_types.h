#ifndef TAC_COMPONENT_TYPES_H
#define TAC_COMPONENT_TYPES_H

#include "error.h"

namespace tac {
namespace internal {

enum class ComponentType {Int, Float, Double};

template <typename T> inline ComponentType typeOf() {
  ierror << "Unsupported type";
  return ComponentType::Double;
}

template <> inline ComponentType typeOf<int>() {
  return ComponentType::Int;
}

template <> inline ComponentType typeOf<float>() {
  return ComponentType::Float;
}

template <> inline ComponentType typeOf<double>() {
  return ComponentType::Double;
}

}}

#endif
