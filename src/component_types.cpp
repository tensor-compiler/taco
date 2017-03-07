#include "component_types.h"

namespace taco {

bool operator==(const ComponentType& a, const ComponentType& b) {
  return a.getKind() == b.getKind();
}

bool operator!=(const ComponentType& a, const ComponentType& b) {
  return a.getKind() != b.getKind();
}

std::ostream& operator<<(std::ostream& os, const ComponentType& type) {
  switch (type.getKind()) {
    case ComponentType::Bool:
      os << "bool";
      break;
    case ComponentType::Int:
      os << "int";
      break;
    case ComponentType::Float:
      os << "float";
      break;
    case ComponentType::Double:
      os << "double";
      break;
    case ComponentType::Unknown:
      break;
  }
  return os;
}

}
