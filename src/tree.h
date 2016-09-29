#ifndef TAC_TREE_H
#define TAC_TREE_H

#include <memory>
#include <ostream>
#include <iostream>

#include "error.h"

namespace tac {

class TreeLevel;

struct Level {
  enum Type {Values, Dense, Sparse};  // TODO: Fixed, Replicated
  Level(Type type) : type(type) {}
  Type type;
};

}
#endif
