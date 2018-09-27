#ifndef TACO_STORAGE_INDEX_H
#define TACO_STORAGE_INDEX_H

#include <memory>
#include <vector>
#include <ostream>

#include "taco/format.h"
#include "taco/taco_tensor_t.h"

namespace taco {
class ModeIndex;
class Array;

/// An index contains the index data structures of a tensor, but not its values.
/// Thus, an index has a format and zero or more mode indices that describes the
/// non-empty coordinates in each mode.
class Index {
public:
  /// Construct an empty index.
  Index();

  /// Construct an index with the given format.
  Index(const Format& format);

  /// Construct an index with the given format and data.
  /// TODO DEPRECATE
  Index(const Format& format, const std::vector<ModeIndex>& indices);

  /// Returns the index's format.
  const Format& getFormat() const;

  /// Returns the number of indices (same as order in format);
  int numModeIndices() const;

  /// Returns the ith mode sub-index.
  /// @{
  const ModeIndex& getModeIndex(int i) const;
  ModeIndex getModeIndex(int i);
  /// @}

  /// Returns the index size, which is the number of values it describes.
  size_t getSize() const;

private:
  struct Content;
  std::shared_ptr<Content> content;
};

std::ostream& operator<<(std::ostream&, const Index&);


/// A mode sub-index of an Index. The type of the mode index is determined by
/// the Format of the Index it is part of.
class ModeIndex {
public:
  /// Construct an empty mode index.
  ModeIndex();

  /// Construct a mode index from a set of index arrays.
  ModeIndex(const std::vector<Array>& indexArrays);

  /// Returns the number of index arrays in this mode index.
  int numIndexArrays() const;

  /// Returns the ith index array. The number of index arrays are dictated by
  /// the ModeIndex's format in its parent Index.
  /// @{
  const Array& getIndexArray(int i) const;
  Array getIndexArray(int i);
  /// @}

private:
  struct Content;
  std::shared_ptr<Content> content;
};


/// Factory functions to construct a compressed sparse rows (CSR) index.
/// @{
Index makeCSRIndex(size_t numrows, int* rowptr, int* colidx);
Index makeCSRIndex(const std::vector<int>& rowptr,
                   const std::vector<int>& colidx);
/// @}

/// Factory functions to construct a compressed sparse columns (CSC) index.
/// @{
Index makeCSCIndex(size_t numrows, int* colptr, int* rowidx);
Index makeCSCIndex(const std::vector<int>& colptr,
                   const std::vector<int>& rowidx);
/// @}

}
#endif
