#ifndef TACO_PACKED_TENSOR_H
#define TACO_PACKED_TENSOR_H

#include <cstdlib>
#include <utility>
#include <vector>
#include <inttypes.h>
#include <ostream>
#include <iostream>
#include <string.h>

#include "format.h"
#include "util/collections.h"
#include "util/strings.h"
#include "util/uncopyable.h"

namespace taco {

/// The index storage for one tree level.
struct LevelStorage {

  void setPtr(const std::vector<int>& ptrVector) {
    this->ptrSize = ptrVector.size();
    this->ptr = util::copyToArray(ptrVector);
  }

  void setIdx(const std::vector<int>& idxVector) {
    this->idxSize = idxVector.size();
    this->idx = util::copyToArray(idxVector);
  }

  // TODO: Remove this function
  std::vector<int> getPtrAsVector() const {
    return util::copyToVector(ptr, ptrSize);
  }

  // TODO: Remove this function
  std::vector<int> getIdxAsVector() const {
    return util::copyToVector(idx, idxSize);
  }

  LevelType levelType;

  // TODO: Free these pointers
  int  ptrSize = 0;
  int* ptr;


  int  idxSize = 0;
  int* idx     = nullptr;
};

class PackedTensor {
public:
  PackedTensor(const std::vector<LevelStorage>& levelStorage,
               const std::vector<double> vals)
      : nnz(vals.size()) {
    values = (double*)malloc(sizeof(double) * nnz);
    memcpy(values, vals.data(), nnz*sizeof(double));

    this->levelStorage = levelStorage;
  }

  size_t getNnz() const {
    return nnz;
  }

  double* getValues() const {
    return values;
  }

  const std::vector<LevelStorage>& getLevelStorage() const {
    return levelStorage;
  }

private:
  int nnz;
  std::vector<LevelStorage> levelStorage;
  double*  values;
};

std::ostream& operator<<(std::ostream& os, const PackedTensor& tp);

}
#endif
