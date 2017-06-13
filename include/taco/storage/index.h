#ifndef TACO_STORAGE_INDEX_H
#define TACO_STORAGE_INDEX_H

#include <memory>
#include <vector>
#include <ostream>

#include "taco/format.h"

namespace taco {

namespace storage {
class DimensionIndex;
class Array;

/// An index contains the index data structures of a tensor, but not its values.
/// Thus, an index has a format and zero or more dimensions indices, describing
/// the non-empty coordinates in each dimension.
class Index {
public:
  /// Construct an empty index.
  Index();

  /// Construct an index with the given format and data.
  Index(const Format& format, const std::vector<DimensionIndex>& indices);

  /// Returns the index's format.
  const Format& getFormat() const;

  /// Returns the ith dimension sub-index.
  const DimensionIndex& getDimensionIndex(int i) const;

  /// Returns the index size, which is the number of values it describes.
  size_t getSize() const;

private:
  struct Content;
  std::shared_ptr<Content> content;
};

std::ostream& operator<<(std::ostream&, const Index&);


/// A dimension sub-index of an Index. The type of the dimension index is
/// determined by the Format of the Index it is part of.
class DimensionIndex {
public:
  /// Construct a dimension index from a set of index arrays.
  DimensionIndex(const std::vector<Array>& indexArrays);

  /// Returns the number of index arrays in this dimension index.
  size_t numIndexArrays() const;

  /// Returns the ith index array. The number of index arrays are dictated by
  /// the DimensionIndex's format in its parent Index.
  const Array& getIndexArray(int i) const;

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

}}
#endif
