#include "taco/storage/storage.h"

#include <iostream>
#include <string>

#include "taco/format.h"
#include "taco/util/error.h"
#include "taco/util/strings.h"

using namespace std;

namespace taco {
namespace storage {

// class Storage::Size
size_t Storage::Size::numValues() const {
  return numVals;
}

size_t
Storage::Size::numIndexValues(size_t dimension, size_t indexNumber) const {
  taco_iassert(dimension < numIndexVals.size());
  taco_iassert(indexNumber < numIndexVals[dimension].size());
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


// class Storage
struct Storage::Content {
  Format                 format;

  vector<LevelIndex>     indices;
  double*                values;

  ~Content() {
    for (auto& indexArray : indices) {
      free(indexArray.ptr);
      free(indexArray.idx);
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
  for (auto& index : content->indices) {
    index.ptr = nullptr;
    index.idx = nullptr;
  }
  content->values = nullptr;
}

void Storage::setFormat(const Format& format) {
  content->format = format;
}

void Storage::setLevelIndex(size_t level, const LevelIndex& index) {
  free(content->indices[level].ptr);
  free(content->indices[level].idx);
  content->indices[level] = index;
}

void Storage::setValues(double* values) {
  free(content->values);
  content->values = values;
}

const Format& Storage::getFormat() const {
  return content->format;
}

const Storage::LevelIndex&
Storage::getLevelIndex(size_t level) const {
  return content->indices[level];
}

Storage::LevelIndex& Storage::getLevelIndex(size_t level) {
  return content->indices[level];
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
    LevelIndex index = content->indices[i];
    if (index.ptr == nullptr) {
      taco_iassert(index.idx == nullptr) << "idx array initialized but not pos";
      continue;
    }

    switch (content->format.getLevels()[i].getType()) {
      case LevelType::Dense:
        numIndexVals[i].push_back(1);  // size
        numVals *= index.ptr[0];
        break;
      case LevelType::Sparse:
        numIndexVals[i].push_back(numVals + 1);         // idx
        numIndexVals[i].push_back(index.ptr[numVals]);  // pos
        numVals = index.ptr[numVals];
        break;
      case LevelType::Fixed:
        numVals *= index.ptr[0];
        numIndexVals[i].push_back(1);
        numIndexVals[i].push_back(numVals);
        break;
    }
  }

  auto size = Storage::Size(numVals, numIndexVals);
  return size;
}

size_t Storage::numBytes() const {
  return getSize().numBytes();
}

std::ostream& operator<<(std::ostream& os, const Storage& storage) {
  auto format = storage.getFormat();
  auto size = storage.getSize();

  // Print indices
  for (size_t i=0; i < format.getLevels().size(); ++i) {
    auto levelIndex = storage.getLevelIndex(i);
//    auto levelSize = size.indexSizes[i];

    os << "d" << to_string(i+1) << ":" << std::endl;
    os << "  ptr: "
       << (levelIndex.ptr != nullptr
           ? "{"+util::join(levelIndex.ptr,
                            levelIndex.ptr + size.numIndexValues(i,0))+"}"
           : "none")
       << std::endl;

    os << "  idx: "
       << (levelIndex.idx != nullptr
           ? "{"+util::join(levelIndex.idx,
                            levelIndex.idx + size.numIndexValues(i,1))+"}"
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

}}
