#include "taco/storage/storage.h"

#include <iostream>
#include <string>

#include "taco/format.h"
#include "taco/error.h"
#include "taco/storage/index.h"
#include "taco/util/strings.h"

using namespace std;

namespace taco {
namespace storage {

// class Storage
struct Storage::Content {
  Format  format;
  Index   index;

  double* values;

  vector<vector<int*>> indices;

  ~Content() {
    for (auto& index : indices) {
      for (auto& indexArray : index) {
        free(indexArray);
      }
    }
    free(values);
  }
};

Storage::Storage() : content(nullptr) {
}

Storage::Storage(const Format& format) : content(new Content) {
  content->format = format;

  auto dimTypes = format.getDimensionTypes();
  content->indices.resize(dimTypes.size());
  for (size_t i = 0; i < content->indices.size(); i++) {
    switch (dimTypes[i]) {
      case DimensionType::Dense:
        content->indices[i].resize(1);
        break;
      case DimensionType::Sparse:
      case DimensionType::Fixed:
        content->indices[i].resize(2);
        break;
    }
    for (size_t j = 0; j < content->indices[i].size(); j++) {
      content->indices[i][j] = nullptr;
    }
  }

  content->values = nullptr;
}

void Storage::setDimensionIndex(size_t dimension, std::vector<int*> index) {
  taco_iassert(index.size() == content->indices[dimension].size()) <<
      "Setting the wrong number of indices (" <<
      index.size() << " != " << content->indices[dimension].size() << "). " <<
      "Type: " << content->format.getDimensionTypes()[dimension] <<
      " (" << content->format.getDimensionOrder()[dimension] << ")";

  for (size_t i = 0; i < content->indices[dimension].size(); i++) {
    content->indices[dimension][i] = index[i];
  }
}

void Storage::setValues(double* values) {
  content->values = values;
} 

const Format& Storage::getFormat() const {
  return content->format;
}

void Storage::setIndex(const Index& index) {
  content->index = index;
}

const Index& Storage::getIndex() const {
  return content->index;
}

Index Storage::getIndex() {
  return content->index;
}

const double* Storage::getValues() const {
  return content->values;
}

double* Storage::getValues() {
  return content->values;
}

Storage::Size Storage::getSize() const {
  vector<vector<size_t>> numIndexVals(content->indices.size());

  size_t numVals = 1;
  for (size_t i=0; i < content->indices.size(); ++i) {
    auto& index = content->indices[i];
    switch (content->format.getDimensionTypes()[i]) {
      case DimensionType::Dense:
        numIndexVals[i].push_back(1);                  // size
        numVals *= index[0][0];
        break;
      case DimensionType::Sparse:
        numIndexVals[i].push_back(numVals + 1);        // pos
        numIndexVals[i].push_back(index[0][numVals]);  // idx
        numVals = index[0][numVals];
        break;
      case DimensionType::Fixed:
        numVals *= index[0][0];
        numIndexVals[i].push_back(1);                  // pos
        numIndexVals[i].push_back(numVals);            // idx
        break;
    }
  }

  return Storage::Size(numVals, numIndexVals);
}

std::ostream& operator<<(std::ostream& os, const Storage& storage) {
  auto format = storage.getFormat();
  if (storage.getValues() == nullptr) {
    return os;
  }

  auto index = storage.getIndex();

  // Print index
  os << index << endl;

  // Print values
  auto values = storage.getValues();
  os << (values != nullptr
         ? "  [" + util::join(values, values + index.getSize()) + "]"
         : "none");

  return os;
}


// class Storage::Size
size_t Storage::Size::numValues() const {
  return numVals;
}

size_t
Storage::Size::numIndexValues(size_t dimension, size_t indexNumber) const {
  taco_iassert(dimension < numIndexVals.size());
  taco_iassert(indexNumber < numIndexVals[dimension].size()) <<
      "not " << indexNumber << " < " << numIndexVals[dimension].size();
  return numIndexVals[dimension][indexNumber];
}

size_t Storage::Size::numBytes() const {
  int cost = numValues() * numBytesPerValue();
  for (size_t i=0; i < numIndexVals.size(); ++i) {
    for (size_t j = 0; j < numIndexVals[i].size(); j++) {
      cost += numIndexValues(i,j) * numBytesPerIndexValue(i,j);
    }
  }
  return cost;
}
size_t Storage::Size::numBytesPerValue() const {
  return sizeof(double);
}

size_t Storage::Size::numBytesPerIndexValue(size_t dim, size_t n) const {
  return sizeof(int);
}

Storage::Size::Size(size_t numVals, vector<vector<size_t>> numIndexVals)
 : numVals(numVals), numIndexVals(numIndexVals) {}

}}
