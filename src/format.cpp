#include "taco/format.h"

#include <iostream>

#include "taco/error.h"
#include "taco/util/strings.h"

namespace taco {

// class Format
Format::Format() {
}

Format::Format(const ModeType& modeType) {
  this->modeTypes.push_back(modeType);
  this->modeOrder.push_back(0);
}

Format::Format(const std::vector<ModeType>& modeTypes) {
  this->modeTypes = modeTypes;
  this->modeOrder.resize(modeTypes.size());
  for (size_t i=0; i < modeTypes.size(); ++i) {
    this->modeOrder[i] = i;
  }
}

Format::Format(const std::vector<ModeType>& modeTypes,
               const std::vector<int>& modeOrder) {
  taco_uassert(modeTypes.size() == modeOrder.size()) <<
      "You must either provide a complete mode ordering or none";
  this->modeTypes = modeTypes;
  this->modeOrder = modeOrder;
}

size_t Format::getOrder() const {
  taco_iassert(this->modeTypes.size() == this->getModeOrder().size());
  return this->modeTypes.size();
}

const std::vector<ModeType>& Format::getModeTypes() const {
  return this->modeTypes;
}

const std::vector<int>& Format::getModeOrder() const {
  return this->modeOrder;
}

bool operator==(const Format& a, const Format& b){
  auto aDimTypes = a.getModeTypes();
  auto bDimTypes = b.getModeTypes();
  auto aDimOrder = a.getModeOrder();
  auto bDimOrder = b.getModeOrder();
  if (aDimTypes.size() == bDimTypes.size()) {
    for (size_t i = 0; i < aDimTypes.size(); i++) {
      if ((aDimTypes[i] != bDimTypes[i]) || (aDimOrder[i] != bDimOrder[i])) {
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
            << util::join(format.getModeOrder(), ",") << ")";
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
