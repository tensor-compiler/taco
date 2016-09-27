#include "expr.h"

#include "util/name_generator.h"

namespace tac {



Var::Var(const std::string& name) : name(name) {
}

Var::Var() : name(util::uniqueName("i")) {
}

}
