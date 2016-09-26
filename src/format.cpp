#include "format.h"

#include <iostream>

#include "tree.h"

using namespace std;

namespace tac {

// Format
Format::Format(std::string format) {
  this->forest = TreeLevel::make(format);
}

std::ostream &operator<<(std::ostream &os, const Format &storage) {
  return os << storage.forest;
}

}
