#ifndef TACO_PACKED_TENSOR_H
#define TACO_PACKED_TENSOR_H

#include <cstdlib>
#include <utility>
#include <vector>
#include <inttypes.h>

namespace taco {

class PackedTensor {
public:
  typedef uint32_t                IndexType;
  typedef std::vector<IndexType>  IndexArray;
  typedef std::vector<IndexArray> Index;
  typedef std::vector<Index>      Indices;

  PackedTensor(size_t nnz, void* values, const Indices& indices)
      : nnz(nnz), values(values), indices(indices) {}

  ~PackedTensor() {
    free(values);
  }

  size_t getNnz() const {
    return nnz;
  }

  const void* getValues() const {
    return values;
  }

  const Indices& getIndices() const {
    return indices;
  }

private:
  size_t nnz;
  void* values;
  Indices indices;
};

}
#endif
