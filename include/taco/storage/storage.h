#ifndef TACO_STORAGE_H
#define TACO_STORAGE_H

#include <vector>
#include <memory>

namespace taco {
class Format;
namespace storage {

/// Storage for a tensor object.  Tensor storage consists of a value array that
/// contains the tensor values and one index per dimension.  The type of each
/// dimension index is determined by the dimension type in the format, and the
/// ordere of the dimension indices is determined by the format dimension order.
class Storage {
public:
  class Size;

  struct LevelIndex {
    LevelIndex() : ptr(nullptr), idx(nullptr) {}
    LevelIndex(int* ptr, int* idx) : ptr(ptr), idx(idx) {}
    int* ptr;
    int* idx;
  };

  /// Construct an undefined tensor storage.
  Storage();

  /// Construct tensor storage for the given format.
  Storage(const Format& format);

  /// Set the format of the tensor storage.  The format describes the indices
  /// of the tensor storage.
  void setFormat(const Format& format);

  /// Set the ith level index.
  void setLevelIndex(size_t level, const LevelIndex& index);

  /// Set the tensor component value array.
  void setValues(double* vals);

  /// Returns the tensor storage format.
  const Format& getFormat() const;

  /// Returns the size of the idx/ptr arrays of each index. The cost of this
  /// function is O(#level).
  Storage::Size getSize() const;

  /// Returns the index for the given level.  The index content is determined
  /// by the level type, which can be read from the format.
  const Storage::LevelIndex& getLevelIndex(size_t level) const;

  /// Returns the index for the given level.  The index content is determined
  /// by the level type, which can be read from the format.
  Storage::LevelIndex& getLevelIndex(size_t level);

  /// Returns the value array that contains the tensor components.
  const double* getValues() const;

  /// Returns the tensor component value array.
  double* getValues();

  /// Storage size
  class Size {
  public:
    /// Returns the number of component values in the tensor storage.
    size_t numValues() const;

    /// Returns the number of values in one of the indices of a given dimension.
    /// The number of indices of each dimension depends on the dimension types
    /// of the tensor storage format.
    size_t numIndexValues(size_t dimension, size_t indexNumber) const;

    /// Returns the total size of the tensor storage.  This includes the size
    /// of indices and component values.
    size_t numBytes() const;

    /// Returns the number of bytes required to store a component.
    size_t numBytesPerValue() const;

    /// Returns the number of bytes required to store a value in one of the
    /// indices of the given dimension.  The number of indices of each dimension
    /// depends on the dimension types of the tensor storage format.
    size_t numBytesPerIndexValue(size_t dimension, size_t indexNumber) const;

  private:
    size_t numVals;
    std::vector<std::vector<size_t>> numIndexVals;

    Size(size_t numVals, std::vector<std::vector<size_t>> numIndexVals);
    friend Storage::Size Storage::getSize() const;
  };

private:
  struct Content;
  std::shared_ptr<Content> content;
};

/// Print Storage objects to a stream.
std::ostream& operator<<(std::ostream&, const Storage&);

}}
#endif
