#include "var.h"

#include <iostream>
#include <ostream>

#include "util/name_generator.h"

namespace taco {

Var::Var(const std::string& name, Kind kind) : content(new Content) {
  content->name = util::uniqueName(name);
  content->kind = kind;
}

Var::Var(Kind kind) : Var("t", kind) {
}

std::ostream& operator<<(std::ostream& os, const Var& var) {
  return os << (var.getKind() == Var::Sum ? "+" : "") << var.getName();
}

}
