/// The pack machinery packs a tensor's non-zero components according to the
/// tensor's storage format.  The machinery is available both as an interpreter
/// that can pack into any format, and as a code generator that generates
/// specialized packing code for one format.

#ifndef TACO_STORAGE_PACK_H
#define TACO_STORAGE_PACK_H

#include <vector>
#include "taco/type.h"
using namespace std;

namespace taco {
  class Format;
  namespace ir {
    class Stmt;
  }
  namespace storage {
    class Storage;

/// Count unique entries (assumes the values are sorted)
vector<int> getUniqueEntries(const vector<int>::const_iterator& begin,
                             const vector<int>::const_iterator& end);

size_t findMaxFixedValue(const vector<int>& dimensions,
                         const vector<vector<int>>& coords,
                         size_t order,
                         const size_t fixedLevel,
                         const size_t i, const size_t numCoords);

Storage pack(const std::vector<int>&              dimensions,
             const Format&                        format,
             const std::vector<std::vector<int>>& coordinates,
             const void *            values,
             const size_t numCoordinates,
             DataType datatype);

/// Generate code to pack tensor coordinates into a specific format. In the
/// generated code the coordinates must be stored as a structure of arrays,
/// that is one vector per axis coordinate and one vector for the values.
/// The coordinates must be sorted lexicographically.
ir::Stmt packCode(const Format& format);

}}
#endif
