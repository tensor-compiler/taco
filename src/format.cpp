#include "format.h"

#include <iostream>

#include "error.h"

using namespace std;

namespace taco {

// class Format
Format::Format() {
}

Format::Format(vector<Level> levels) : levels(levels) {
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
  for (auto& level : format.getLevels()) {
    switch (level.getType()) {
      case LevelType::Dense:
        os << "d";
        break;
      case LevelType::Sparse:
        os << "s";
        break;
      case LevelType::Fixed:
        os << "f";
        break;
      case LevelType::Repeated:
        os << "r";
        break;
      case LevelType::Replicated:
        os << "p";
        break;
    }
  }
  return os;
}

}
