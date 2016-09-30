#ifndef TACO_PACKED_TENSOR_H
#define TACO_PACKED_TENSOR_H

#include <vector>
#include <cstdlib>
#include <inttypes.h>

namespace taco {

class PackedTensor {
public:
  typedef uint32_t IndexType;

  PackedTensor(size_t nnz, void* values,
               const std::vector<std::vector<IndexType*>>& indices)
      : nnz(nnz), values(values), indices(indices) {}

  ~PackedTensor() {
    free(values);
    for (auto& index : indices) {
      for (auto& indexArray : index) {
        free(indexArray);
      }
    }
  }

  size_t getNnz() const {
    return nnz;
  }

  const void* getValues() const {
    return values;
  }

  const std::vector<std::vector<IndexType*>>& getIndices() const {
    return indices;
  }

private:
  size_t nnz;
  void* values;
  std::vector<std::vector<IndexType*>> indices;
};

}
#endif
