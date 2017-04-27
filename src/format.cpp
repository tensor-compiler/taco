#include "taco/format.h"

#include <iostream>

#include "taco/util/error.h"
#include "taco/util/strings.h"

namespace taco {

const Format DVEC({Dense});
const Format SVEC({Sparse});

const Format DMAT({Dense, Dense},{0,1});
const Format CSR({Dense, Sparse},{0,1});
const Format CSC({Dense, Sparse},{1,0});
const Format ELL({Dense, Fixed},{0,1});

// class Format
Format::Format() {
}

Format::Format(const std::vector<LevelType>& levelTypes,
               const std::vector<size_t>& dimensionOrder) {
  taco_uassert(levelTypes.size() == dimensionOrder.size())
      << "You must either provide a complete dimension ordering or none";
  for (size_t i=0; i < levelTypes.size(); ++i) {
    levels.push_back(Level(dimensionOrder[i], levelTypes[i]));
  }
}

Format::Format(const std::vector<LevelType>& levelTypes) {
  for (size_t i=0; i < levelTypes.size(); ++i) {
    levels.push_back(Level(i, levelTypes[i]));
  }
}

bool operator==(const Format& l, const Format& r){
  auto& llevels = l.getLevels();
  auto& rlevels = r.getLevels();
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
