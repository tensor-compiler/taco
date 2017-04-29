#include "taco/storage/storage.h"

#include <iostream>
#include <string>

#include "taco/format.h"
#include "taco/util/error.h"
#include "taco/util/strings.h"

using namespace std;

namespace taco {
namespace storage {

// class TensorTensorStorage
struct Storage::Content {
  Format             format;
  vector<LevelIndex> index;
  double*            values;

  ~Content() {
    for (auto& indexArray : index) {
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
  content->index.resize(levels.size());
  for (auto& index : content->index) {
    index.ptr = nullptr;
    index.idx = nullptr;
  }
  content->values = nullptr;
}

void Storage::setFormat(const Format& format) {
  content->format = format;
}

void Storage::setLevelIndex(size_t level, const LevelIndex& index) {
  free(content->index[level].ptr);
  free(content->index[level].idx);
  content->index[level] = index;
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
  return content->index[level];
}

Storage::LevelIndex& Storage::getLevelIndex(size_t level) {
  return content->index[level];
}

const double* Storage::getValues() const {
  return content->values;
}

double* Storage::getValues() {
  return content->values;
}

Storage::Size Storage::getSize() const {
  Storage::Size size;
  int numLevels = (int)content->index.size();

  size.indexSizes.resize(numLevels);
  size_t prevIdxSize = 1;
  for (size_t i=0; i < content->index.size(); ++i) {
    LevelIndex index = content->index[i];
    taco_iassert(index.ptr != nullptr) << "Index not allocated";
    switch (content->format.getLevels()[i].getType()) {
      case LevelType::Dense:
        size.indexSizes[i].ptr = 1;
        size.indexSizes[i].idx = 0;
        prevIdxSize *= index.ptr[0];
        break;
      case LevelType::Sparse:
        size.indexSizes[i].ptr = prevIdxSize + 1;
        size.indexSizes[i].idx = index.ptr[prevIdxSize];
        prevIdxSize = index.ptr[prevIdxSize];
        break;
      case LevelType::Fixed:
        size.indexSizes[i].ptr = 1;
        prevIdxSize *= index.ptr[0];
        size.indexSizes[i].idx = prevIdxSize;
        break;
    }
  }
  size.values = prevIdxSize;
  return size;
}

int Storage::numBytes() const {
  Storage::Size size = getSize();
  int cost = size.values*sizeof(double);
  for (size_t i=0; i < content->index.size(); ++i) {
    cost += size.indexSizes[i].idx*sizeof(int);
    cost += size.indexSizes[i].ptr*sizeof(int);
  }
  return cost;
}

bool Storage::defined() const {
  return content != nullptr;
}

std::ostream& operator<<(std::ostream& os, const Storage& storage) {
  auto format = storage.getFormat();
  auto size = storage.getSize();

  // Print indices
  for (size_t i=0; i < format.getLevels().size(); ++i) {
    auto levelIndex = storage.getLevelIndex(i);
    auto levelSize = size.indexSizes[i];

    os << "d" << to_string(i+1) << ":" << std::endl;
    os << "  ptr: "
       << (levelIndex.ptr != nullptr
           ? "{"+util::join(levelIndex.ptr, levelIndex.ptr+levelSize.ptr)+"}"
           : "none")
       << std::endl;

    os << "  idx: "
       << (levelIndex.idx != nullptr
           ? "{"+util::join(levelIndex.idx, levelIndex.idx+levelSize.idx)+"}"
           : "none")
       << std::endl;
  }

  // Print values
  auto values = storage.getValues();
  os << "vals:  "
     << (values != nullptr
         ? "{"+util::join(values, values+size.values)+"}"
         : "none");

  return os;
}

}}
