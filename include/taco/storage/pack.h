/// The pack machinery packs a tensor's non-zero components according to the
/// tensor's storage format.  The machinery is available both as an interpreter
/// that can pack into any format, and as a code generator that generates
/// specialized packing code for one format.

#ifndef TACO_STORAGE_PACK_H
#define TACO_STORAGE_PACK_H

#include <climits>
#include <vector>

#include "taco/type.h"
#include "taco/format.h"
#include "taco/storage/typed_vector.h"
#include "taco/storage/storage.h"
#include "taco/storage/coordinate.h"
 
namespace taco {

namespace ir {
class Stmt;
}

TensorStorage pack(Datatype                             datatype,
                   const std::vector<int>&              dimensions,
                   const Format&                        format,
                   const std::vector<TypedIndexVector>& coordinates,
                   const void*                          values);


template<typename V, size_t O, typename C>
TensorStorage pack(std::vector<int> dimensions, Format format,
                   const std::vector<std::pair<Coordinates<O,C>,V>>& components){
  size_t order = dimensions.size();
  size_t nnz = components.size();

  std::vector<TypedIndexVector> coordinates(order,
                                            TypedIndexVector(type<C>(), nnz));
  std::vector<V> values(nnz);
  for (size_t i = 0; i < nnz; i++) {
    values[i] = components[i].second;
    auto& coords = components[i].first;
    for (size_t j = 0; j < order; j++) {
      coordinates[j][i] = coords[j];
    }
  }

  return pack(type<V>(), dimensions, format, coordinates, values.data());
}

}
#endif
