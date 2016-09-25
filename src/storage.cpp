#include "storage.h"

#include <iostream>

#include "error.h"

using namespace std;

namespace tac {

// TensorStorage
TensorStorage::TensorStorage(std::string format) {
  this->forest = TreeLevel::make(format);
}

std::ostream &operator<<(std::ostream &os, const TensorStorage &storage) {
  return os << storage.forest;
}


// TreeLevel
TreeLevelPtr TreeLevel::make(std::string format) {
  TreeLevelPtr level = values();
  for (size_t i=0; i < format.size(); ++i) {
    switch (format[i]) {
      case 'd':
        level = dense(level);
        break;
      case 's':
        level = sparse(level);
        break;
      case 'f':
        level = fixed(level);
        break;
      case 'r':
        level = replicated(level);
        break;
      default:
        uerror << "Format character not recognized: " << format[i];
    }
  }
  return level;
}

const TreeLevelPtr& TreeLevel::getChildren() const {
  return this->subLevel;
}


// Factory functions
TreeLevelPtr values() {
  return shared_ptr<TreeLevel>(new Values());
}

TreeLevelPtr dense(const TreeLevelPtr& subLevel) {
  return shared_ptr<TreeLevel>(new Dense(subLevel));
}

TreeLevelPtr sparse(const TreeLevelPtr& subLevel) {
  return shared_ptr<TreeLevel>(new Sparse(subLevel));
}

TreeLevelPtr fixed(const TreeLevelPtr& subLevel) {
  return shared_ptr<TreeLevel>(new Fixed(subLevel));
}

TreeLevelPtr replicated(const TreeLevelPtr& subLevel) {
  return shared_ptr<TreeLevel>(new Replicated(subLevel));
}

}
