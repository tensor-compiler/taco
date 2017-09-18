/// The pack machinery packs a tensor's non-zero components according to the
/// tensor's storage format.  The machinery is available both as an interpreter
/// that can pack into any format, and as a code generator that generates
/// specialized packing code for one format.

#ifndef TACO_STORAGE_PACK_H
#define TACO_STORAGE_PACK_H

#include <vector>

namespace taco {
class Format;
namespace ir {
class Stmt;
}
namespace storage {
class Storage;

/// Pack tensor coordinates into a format. The coordinates must be stored as a
/// structure of arrays, that is one vector per axis coordinate and one vector
/// for the values. The coordinates must be sorted lexicographically.
Storage pack(const std::vector<int>&              dimensions,
             const Format&                        format,
             const std::vector<std::vector<int>>& coordinates,
             const std::vector<double>            values);

/// Generate code to pack tensor coordinates into a specific format. In the
/// generated code the coordinates must be stored as a structure of arrays,
/// that is one vector per axis coordinate and one vector for the values.
/// The coordinates must be sorted lexicographically.
ir::Stmt packCode(const Format& format);

}}
#endif
