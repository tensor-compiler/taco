#ifndef TACO_TREE_H
#define TACO_TREE_H

#include <memory>
#include <ostream>
#include <iostream>

#include "error.h"

namespace taco {

struct Level {
  enum Type {Values, Dense, Sparse};  // TODO: Fixed, Replicated
  Level(Type type) : type(type) {}
  Type type;
};

}
#endif
