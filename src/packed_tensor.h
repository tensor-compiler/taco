#ifndef TACO_PACKED_TENSOR_H
#define TACO_PACKED_TENSOR_H

#include <cstdlib>
#include <utility>
#include <vector>
#include <inttypes.h>

namespace taco {

class PackedTensor {
public:
  // TODO: Change all these types to void pointers to support multiple
  //       index/value types
  typedef uint32_t                IndexType;
  typedef std::vector<IndexType>  IndexArray;
  typedef std::vector<IndexArray> Index;
  typedef std::vector<Index>      Indices;
  typedef std::vector<double>     Values;

  PackedTensor(size_t nnz, const Values& values, const Indices& indices)
      : nnz(nnz), values(values), indices(indices) {}

  size_t getNnz() const {
    return nnz;
  }

  const Values& getValues() const {
    return values;
  }

  const Indices& getIndices() const {
    return indices;
  }

private:
  size_t nnz;
  Values  values;
  Indices indices;
};

}
#endif
