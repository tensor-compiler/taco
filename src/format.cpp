#include "taco/format.h"

#include <iostream>
#include <climits>

#include "taco/error.h"
#include "taco/util/strings.h"

namespace taco {

// class Format
Format::Format() {
}

Format::Format(const ModeType& modeType) {
  this->modeTypes.push_back(modeType);
  this->modeOrdering.push_back(0);
}

Format::Format(const std::vector<ModeType>& modeTypes) {
  this->modeTypes = modeTypes;
  this->modeOrdering.resize(modeTypes.size());
  taco_uassert(modeTypes.size() <= INT_MAX) << "Supports only INT_MAX modes";
  for (int i=0; i < static_cast<int>(modeTypes.size()); ++i) {
    this->modeOrdering[i] = i;
  }
}

Format::Format(const std::vector<ModeType>& modeTypes,
               const std::vector<size_t>& modeOrdering) {
  taco_uassert(modeTypes.size() == modeOrdering.size()) <<
      "You must either provide a complete mode ordering or none";
  this->modeTypes = modeTypes;
  this->modeOrdering = modeOrdering;
}

size_t Format::getOrder() const {
  taco_iassert(this->modeTypes.size() == this->getModeOrdering().size());
  return this->modeTypes.size();
}

const std::vector<ModeType>& Format::getModeTypes() const {
  return this->modeTypes;
}

const std::vector<size_t>& Format::getModeOrdering() const {
  return this->modeOrdering;
}

bool operator==(const Format& a, const Format& b){
  auto aModeTypes = a.getModeTypes();
  auto bModeTypes = b.getModeTypes();
  auto aModeOrdering = a.getModeOrdering();
  auto bModeOrdering = b.getModeOrdering();
  if (aModeTypes.size() == bModeTypes.size()) {
    for (size_t i = 0; i < aModeTypes.size(); i++) {
      if ((aModeTypes[i] != bModeTypes[i]) ||
          (aModeOrdering[i] != bModeOrdering[i])) {
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
  return os << "(" << util::join(format.getModeTypes(), ",") << "; "
            << util::join(format.getModeOrdering(), ",") << ")";
}

std::ostream& operator<<(std::ostream& os, const ModeType& modeType) {
  switch (modeType) {
    case ModeType::Dense:
      os << "dense";
      break;
    case ModeType::Sparse:
      os << "sparse";
      break;
    case ModeType::Fixed:
      os << "fixed";
      break;
  }
  return os;
}

// Predefined formats
const Format CSR({Dense, Sparse}, {0,1});
const Format CSC({Dense, Sparse}, {1,0});
const Format DCSR({Sparse, Sparse}, {0,1});
const Format DCSC({Sparse, Sparse}, {1,0});

bool isDense(const Format& format) {
  for (ModeType modeType : format.getModeTypes()) {
    if (modeType != Dense) {
      return false;
    }
  }
  return true;
}

}
