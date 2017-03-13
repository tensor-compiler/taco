#ifndef TACO_COMPONENT_TYPES_H
#define TACO_COMPONENT_TYPES_H

#include <climits>
#include <ostream>

#include "taco/util/error.h"

namespace taco {

class ComponentType {
public:
  enum Kind {Bool, Int, Float, Double, Unknown};

  ComponentType() : ComponentType(Unknown) {}
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
      case Unknown:
        break;
    }
    return UINT_MAX;
  }

  Kind getKind() const {
    return kind;
  }

private:
  Kind kind;
};

bool operator==(const ComponentType& a, const ComponentType& b);
bool operator!=(const ComponentType& a, const ComponentType& b);

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

}
#endif
