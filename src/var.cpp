#include "var.h"

#include <ostream>

namespace taco {

std::ostream& operator<<(std::ostream& os, const Var& var) {
  return os << (var.getKind() == Var::Reduction ? "+" : "") << var.getName();
}

}
