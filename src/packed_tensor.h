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
#include "util/strings.h"

namespace taco {

struct LevelStorage {
  // TODO: remove
  LevelType levelType;

  // TODO: replace with pointers
  std::vector<int> ptr;
  std::vector<int> idx;
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
