#include "format.h"

#include <iostream>

#include "util/error.h"
#include "util/strings.h"

namespace taco {

const Format DVEC({Dense});
const Format SVEC({Sparse});
const Format CSR({Dense, Sparse},{0,1});
const Format CSC({Dense, Sparse},{1,0});
const Format DIA({Dense, Offset},{0,1});
const Format ELL({Dense, Fixed},{0,1});
const Format COO({Dense, Replicated},{0,1});

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

bool operator==(const Format& l, const Format& r){
  if (l.levels.size()==r.levels.size()) {
    for (size_t i=0; i<l.levels.size(); i++) {
      if ((l.levels[i].getType()!=r.levels[i].getType()) ||
	  (l.levels[i].getDimension()!=r.levels[i].getDimension())) {
	return false;
      }
    }
    return true;
  }
  return false;
}

bool Format::isCSR() const {
  return (*this == CSR);
}

bool Format::isCSC() const {
  return (*this == CSC);
}

bool Format::isDIA() const {
  return (*this == DIA);
}

bool Format::isELL() const {
  return (*this == ELL);
}

bool Format::isCOO() const {
  return (*this == COO);
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
    case LevelType::Offset:
      os << "offseted";
      break;
    case LevelType::Fixed:
      os << "fixed";
      break;
    case LevelType::Replicated:
      os << "replicated";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const Level& level) {
  return os << level.getDimension() << ":" << level.getType();
}

}
