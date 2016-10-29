#ifndef TACO_PACKED_TENSOR_H
#define TACO_PACKED_TENSOR_H

#include <cstdlib>
#include <utility>
#include <vector>
#include <inttypes.h>
#include <ostream>
#include <iostream>
#include <string.h>
#include <memory>

#include "format.h"
#include "util/collections.h"
#include "util/strings.h"
#include "util/uncopyable.h"

namespace taco {

/// The index storage for one tree level.
class LevelStorage {
public:
  LevelStorage(LevelType levelType);

  /// Set the level storage's ptr. This pointer will be freed by level storage.
  void setPtr(int* ptr);

  /// Set the level storage's idx. This pointer will be freed by level storage.
  void setIdx(int* idx);

  void setPtr(const std::vector<int>& ptrVector);
  void setIdx(const std::vector<int>& idxVector);

  // TODO: Remove these functions
  std::vector<int> getPtrAsVector() const;
  std::vector<int> getIdxAsVector() const;

  int getPtrSize() const;
  int getIdxSize() const;

  const int* getPtr() const;
  const int* getIdx() const;

  int* getPtr();
  int* getIdx();

private:
  struct Content;
  std::shared_ptr<Content> content;
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
