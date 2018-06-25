/// The pack machinery packs a tensor's non-zero components according to the
/// tensor's storage format.  The machinery is available both as an interpreter
/// that can pack into any format, and as a code generator that generates
/// specialized packing code for one format.

#ifndef TACO_STORAGE_PACK_H
#define TACO_STORAGE_PACK_H

#include <climits>
#include <vector>
#include "taco/type.h"
#include "taco/storage/typed_vector.h"

using namespace std;
 
namespace taco {
class Format;
namespace ir {
class Stmt;
}
namespace storage {
class TensorStorage;

TensorStorage pack(DataType                             datatype,
                   const std::vector<int>&              dimensions,
                   const Format&                        format,
                   const std::vector<TypedIndexVector>& coordinates,
                   const void*                          values,
                   size_t                               numCoordinates);

/// Generate code to pack tensor coordinates into a specific format. In the
/// generated code the coordinates must be stored as a structure of arrays,
/// that is one vector per axis coordinate and one vector for the values.
/// The coordinates must be sorted lexicographically.
ir::Stmt packCode(const Format& format);

}}
#endif
