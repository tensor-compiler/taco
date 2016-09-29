#ifndef TAC_COMPONENT_TYPES_H
#define TAC_COMPONENT_TYPES_H

#include "error.h"

namespace tac {
namespace internal {

class ComponentType {
public:
  enum Kind {Int, Float, Double};
  ComponentType(Kind kind) : kind(kind)  {}

  size_t bytes() {
    switch (this->kind) {
      case Int:
        return sizeof(int);
      case Float:
        return sizeof(float);
      case Double:
        return sizeof(double);
    }
    return UINT_MAX;
  }

private:
  Kind kind;
};

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
