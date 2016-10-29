#include "packed_tensor.h"

#include <iostream>
#include <string>

#include "util/strings.h"

using namespace std;

namespace taco {

// class LevelStorage
struct LevelStorage::Content {

  LevelType levelType;

  int  ptrSize = 0;
  int  idxSize = 0;

  int* ptr = nullptr;
  int* idx = nullptr;

  ~Content() {
    free(ptr);
    free(idx);
  }
};

LevelStorage::LevelStorage(LevelType levelType) : content(new Content) {
  content->levelType = levelType;
}

void LevelStorage::setPtr(int* ptr) {
  delete content->ptr;
  content->ptr = ptr;
}

void LevelStorage::setIdx(int* idx) {
  delete content->idx;
  content->idx = idx;
}

void LevelStorage::setPtr(const std::vector<int>& ptrVector) {  
  content->ptrSize = ptrVector.size();
  content->ptr = util::copyToArray(ptrVector);
}

void LevelStorage::setIdx(const std::vector<int>& idxVector) {
  content->idxSize = idxVector.size();
  content->idx = util::copyToArray(idxVector);
}

std::vector<int> LevelStorage::getPtrAsVector() const {
  return util::copyToVector(content->ptr, content->ptrSize);
}

std::vector<int> LevelStorage::getIdxAsVector() const {
  return util::copyToVector(content->idx, content->idxSize);
}

int LevelStorage::getPtrSize() const {
  return content->ptrSize;
}

int LevelStorage::getIdxSize() const {
  return content->idxSize;
}

const int* LevelStorage::getPtr() const {
  return content->ptr;
}

const int* LevelStorage::getIdx() const {
  return content->idx;
}

int* LevelStorage::getPtr() {
  return content->ptr;
}

int* LevelStorage::getIdx() {
  return content->idx;
}


std::ostream& operator<<(std::ostream& os, const PackedTensor& tp) {
  auto& levelStorage = tp.getLevelStorage();
  double* values = tp.getValues();
  auto nnz       = tp.getNnz();

  // Print indices
  for (size_t i=0; i < levelStorage.size(); ++i) {
    auto& level = levelStorage[i];
    os << "L" << to_string(i) << ":" << std::endl;
    os << "  ptr: {" << util::join(level.getPtrAsVector()) << "}" << std::endl;
    os << "  idx: {" << util::join(level.getIdxAsVector()) << "}" << std::endl;
  }

  //  // Print values
  os << "vals:  {" << util::join(values, values+nnz) << "}";

  return os;
}

}
