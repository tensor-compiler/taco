#include "format.h"

#include <iostream>

#include "error.h"
#include "util/strings.h"

namespace taco {

// class Format
Format::Format() {
}

Format::Format(LevelTypes levelTypes, DimensionOrders dimensionOrder) {
  uassert(levelTypes.size()==dimensionOrder.size())
      << "You must either provide a complete dimension ordering or none";
  for (size_t i=0; i < levelTypes.size(); ++i) {
    levels.push_back(Level(dimensionOrder[i], levelTypes[i]));
  }
}

Format::Format(LevelTypes levelTypes) {
  for (size_t i=0; i < levelTypes.size(); ++i) {
    levels.push_back(Level(i, levelTypes[i]));
  }
}

bool Format::isCSR() const {
  if (levels.size()==2) {
    if ((levels[0].getType()==Dense) &&	(levels[1].getType()==Sparse) &&
	(levels[0].getDimension()==0)) {
      return true;
    }
  }
  return false;
}

bool Format::isCSC() const {
  if (levels.size()==2) {
    if ((levels[0].getType()==Dense) &&	(levels[1].getType()==Sparse) &&
	(levels[0].getDimension()==1)) {
      return true;
    }
  }
  return false;
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
