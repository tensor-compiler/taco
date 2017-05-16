#include "taco/format.h"

#include <iostream>

#include "taco/error.h"
#include "taco/util/strings.h"

namespace taco {

// class Format
Format::Format() {
}

Format::Format(const DimensionType& dimensionType) {
  this->dimensionTypes.push_back(dimensionType);
  this->dimensionOrder.push_back(0);
}

Format::Format(const std::vector<DimensionType>& dimensionTypes) {
  this->dimensionTypes = dimensionTypes;
  this->dimensionOrder.resize(dimensionTypes.size());
  for (size_t i=0; i < dimensionTypes.size(); ++i) {
    this->dimensionOrder[i] = i;
  }
}

Format::Format(const std::vector<DimensionType>& dimensionTypes,
               const std::vector<int>& dimensionOrder) {
  taco_uassert(dimensionTypes.size() == dimensionOrder.size()) <<
      "You must either provide a complete dimension ordering or none";
  this->dimensionTypes = dimensionTypes;
  this->dimensionOrder = dimensionOrder;
}

size_t Format::getOrder() const {
  taco_iassert(this->dimensionTypes.size() == this->getDimensionOrder().size());
  return this->dimensionTypes.size();
}

const std::vector<DimensionType>& Format::getDimensionTypes() const {
  return this->dimensionTypes;
}

const std::vector<int>& Format::getDimensionOrder() const {
  return this->dimensionOrder;
}

bool operator==(const Format& a, const Format& b){
  auto aDimTypes = a.getDimensionTypes();
  auto bDimTypes = b.getDimensionTypes();
  auto aDimOrder = a.getDimensionOrder();
  auto bDimOrder = b.getDimensionOrder();
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
  return os << "(" << util::join(format.getDimensionTypes(), ",") << "; "
            << util::join(format.getDimensionOrder(), ",") << ")";
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

// Predefined formats
const Format CSR({Dense, Sparse}, {0,1});
const Format CSC({Dense, Sparse}, {1,0});
const Format DCSR({Sparse, Sparse}, {0,1});
const Format DCSC({Sparse, Sparse}, {1,0});

bool isDense(const Format& format) {
  for (DimensionType dimensionType : format.getDimensionTypes()) {
    if (dimensionType != Dense) {
      return false;
    }
  }
  return true;
}

}
