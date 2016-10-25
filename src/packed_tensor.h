#ifndef TACO_PACKED_TENSOR_H
#define TACO_PACKED_TENSOR_H

#include <cstdlib>
#include <utility>
#include <vector>
#include <inttypes.h>
#include <ostream>


namespace taco {

class PackedTensor {
public:
  // TODO: Change all these types to void pointers to support multiple
  //       index/value types
  typedef int                     IndexType;
//  typedef IndexType*              IndexArray;
  typedef std::vector<IndexType>  IndexArray; // Index values
//  typedef std::pair<IndexArray> Index;      // 2 index arrays per Index
  typedef std::vector<IndexArray> Index;      // [0,2] index arrays per Index
  typedef std::vector<Index>      Indices;    // One Index per level
  //typedef std::vector<double>     Values;
  typedef double*                 Values;
  int                             nnz;        // number of values

  PackedTensor(const Values& values, const Indices& indices)
      : values(values), indices(indices) {}
 
   PackedTensor(const std::vector<double> vals, const Indices& indices)
      : nnz(vals.size()), indices(indices) {
    values = (Values)malloc(sizeof(double) * nnz);
    memcpy(values, vals.data(), nnz*sizeof(double));
   }

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
  Values  values;
  Indices indices;
};

std::ostream& operator<<(std::ostream& os, const PackedTensor& tp);

}
#endif
