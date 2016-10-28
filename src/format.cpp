#include "format.h"

#include <iostream>

#include "error.h"
#include "util/strings.h"

using namespace std;

namespace taco {

// class Format
Format::Format() {
}

Format::Format(vector<LevelType> levelTypes, vector<size_t> dimensionOrder) {
  uassert(levelTypes.size()==dimensionOrder.size())
      << "You must either provide a complete dimension ordering or none";
  for (size_t i=0; i < levelTypes.size(); ++i) {
    levels.push_back(Level(dimensionOrder[i], levelTypes[i]));
  }
}

Format::Format(std::vector<LevelType> levelTypes) {
  for (size_t i=0; i < levelTypes.size(); ++i) {
    levels.push_back(Level(i, levelTypes[i]));
  }
}

std::ostream &operator<<(std::ostream& os, const Format& format) {
  return os << "(" << util::join(format.getLevels()) << ")";
}

std::ostream& operator<<(std::ostream& os, const LevelType& levelType) {
  switch (levelType) {
    case LevelType::Dense:
      os << "dense";
      break;
    case LevelType::Sparse:
      os << "sparse";
      break;
    case LevelType::Fixed:
      os << "fixed";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const Level& level) {
  return os << level.getDimension() << ":" << level.getType();
}

}
