#include "taco/storage/storage.h"

#include <iostream>
#include <string>

#include "taco/format.h"
#include "taco/util/error.h"
#include "taco/util/strings.h"

using namespace std;

namespace taco {
namespace storage {

// class Storage
struct Storage::Content {
  Format               format;

  vector<vector<int*>> indices;
  double*              values;

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
  vector<Level> levels = format.getLevels();
  content->indices.resize(levels.size());
  for (size_t i = 0; i < content->indices.size(); i++) {
    switch (levels[i].getType()) {
      case LevelType::Dense:
        content->indices[i].resize(1);
        break;
      case LevelType::Sparse:
      case LevelType::Fixed:
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
      "Type: " << content->format.getLevels()[dimension];

  for (size_t i = 0; i < content->indices[dimension].size(); i++) {
    content->indices[dimension][i] = index[i];
  }
}

void Storage::setValues(double* values) {
  free(content->values);
  content->values = values;
}

const Format& Storage::getFormat() const {
  return content->format;
}

const vector<int*>& Storage::getDimensionIndex(size_t dimension) const {
  return content->indices[dimension];
}

const double* Storage::getValues() const {
  return content->values;
}

double* Storage::getValues() {
  return content->values;
}

Storage::Size Storage::getSize() const {
  vector<vector<size_t>> numIndexVals;

  numIndexVals.resize(content->indices.size());

  size_t numVals = 1;
  for (size_t i=0; i < content->indices.size(); ++i) {
    auto& index = content->indices[i];
    switch (content->format.getLevels()[i].getType()) {
      case LevelType::Dense:
        numIndexVals[i].push_back(1);                  // size
        numVals *= index[0][0];
        break;
      case LevelType::Sparse:
        numIndexVals[i].push_back(numVals + 1);        // pos
        numIndexVals[i].push_back(index[0][numVals]);  // idx
        numVals = index[0][numVals];
        break;
      case LevelType::Fixed:
        numVals *= index[0][0];
        numIndexVals[i].push_back(1);                  // pos
        numIndexVals[i].push_back(numVals);            // idx
        break;
    }
  }

  auto size = Storage::Size(numVals, numIndexVals);
  return size;
}

std::ostream& operator<<(std::ostream& os, const Storage& storage) {
  auto format = storage.getFormat();
  auto size = storage.getSize();

  // Print indices
  for (size_t i=0; i < format.getLevels().size(); ++i) {
    auto pos = storage.getDimensionIndex(i)[0];
    auto idx = storage.getDimensionIndex(i)[1];

    os << "d" << to_string(i+1) << ":" << std::endl;
    os << "  ptr: "
       << (pos != nullptr
           ? "{"+util::join(pos, pos + size.numIndexValues(i,0))+"}"
           : "none")
       << std::endl;

    os << "  idx: "
       << (idx != nullptr
           ? "{"+util::join(idx, idx + size.numIndexValues(i,1))+"}"
           : "none")
       << std::endl;
  }

  // Print values
  auto values = storage.getValues();
  os << "vals:  "
     << (values != nullptr
         ? "{"+util::join(values, values + size.numValues())+"}"
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
