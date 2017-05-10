#include "taco/format.h"

#include <iostream>

#include "taco/error.h"
#include "taco/util/strings.h"

namespace taco {

// class Format
Format::Format() {
}

Format::Format(const DimensionType& dimensionType) {
  levels.push_back(Level(0, dimensionType));
}

Format::Format(const std::vector<DimensionType>& dimensionTypes) {
  for (size_t i=0; i < dimensionTypes.size(); ++i) {
    levels.push_back(Level(i, dimensionTypes[i]));
  }
}

Format::Format(const std::vector<DimensionType>& dimensionTypes,
               const std::vector<int>& dimensionOrder) {
  taco_uassert(dimensionTypes.size() == dimensionOrder.size()) <<
      "You must either provide a complete dimension ordering or none";
  for (size_t i=0; i < dimensionTypes.size(); ++i) {
    levels.push_back(Level(dimensionOrder[i], dimensionTypes[i]));
  }
}

bool operator==(const Format& a, const Format& b){
  auto& llevels = a.getLevels();
  auto& rlevels = b.getLevels();
  if (llevels.size() == rlevels.size()) {
    for (size_t i = 0; i < llevels.size(); i++) {
      if ((llevels[i].getType() != rlevels[i].getType()) ||
          (llevels[i].getDimension() != rlevels[i].getDimension())) {
        return false;
      }
    }
    return true;
  }
  return false;
}

bool operator!=(const Format& a, const Format& b) {
  return !(a == b);
}

std::ostream &operator<<(std::ostream& os, const Format& format) {
  return os << "(" << util::join(format.getLevels()) << ")";
}

std::ostream& operator<<(std::ostream& os, const DimensionType& dimensionType) {
  switch (dimensionType) {
    case DimensionType::Dense:
      os << "dense";
      break;
    case DimensionType::Sparse:
      os << "sparse";
      break;
    case DimensionType::Fixed:
      os << "fixed";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const Level& level) {
  return os << level.getDimension() << ":" << level.getType();
}

// Predefined formats
const Format CSR({Dense, Sparse}, {0,1});
const Format CSC({Dense, Sparse}, {1,0});
const Format DCSR({Sparse, Sparse}, {0,1});
const Format DCSC({Sparse, Sparse}, {1,0});

}
