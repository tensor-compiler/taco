#ifndef TACO_COMPONENT_TYPES_H
#define TACO_COMPONENT_TYPES_H

#include "error.h"
#include "limits.h"

namespace taco {
namespace internal {

class ComponentType {
public:
  enum Kind {Bool, Int, Float, Double};
  ComponentType(Kind kind) : kind(kind)  {}

  size_t bytes() {
    switch (this->kind) {
      case Bool:
        return sizeof(bool);
      case Int:
        return sizeof(int);
      case Float:
        return sizeof(float);
      case Double:
        return sizeof(double);
    }
    return UINT_MAX;
  }

  Kind getKind() const {
    return kind;
  }

  /** Compare two types for equality */
  bool operator==(const ComponentType &other) const {
    return kind == other.kind;
  }

  /** Compare two types for inequality */
  bool operator!=(const ComponentType &other) const {
    return kind != other.kind;
  }

private:
  Kind kind;
};

std::ostream& operator<<(std::ostream&, const ComponentType&);

template <typename T> inline ComponentType typeOf() {
  ierror << "Unsupported type";
  return ComponentType::Double;
}

template <> inline ComponentType typeOf<bool>() {
  return ComponentType::Bool;
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
