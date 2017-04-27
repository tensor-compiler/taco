#ifndef TACO_STORAGE_H
#define TACO_STORAGE_H

#include <vector>
#include <memory>

namespace taco {
class Format;
namespace storage {

class TensorStorage {
public:
  struct Size {
    struct LevelIndexSize {
      size_t ptr;
      size_t idx;
    };
    std::vector<LevelIndexSize> indexSizes;
    size_t values;
  };

  struct LevelIndex {
    LevelIndex() : ptr(nullptr), idx(nullptr) {}
    LevelIndex(int* ptr, int* idx) : ptr(ptr), idx(idx) {}
    int* ptr;
    int* idx;
  };

  /// Construct an undefined tensor storage.
  TensorStorage();

  /// Construct tensor storage for the given format.
  TensorStorage(const Format& format);

  /// Set the format of the tensor storage.  The format describes the indices
  /// of the tensor storage.
  void setFormat(const Format& format);

  /// Set the ith level index.
  void setLevelIndex(size_t level, const LevelIndex& index);

  /// Set the tensor component value array.
  void setValues(double* vals);

  /// Returns the storage's format.
  const Format& getFormat() const;

  /// Returns the size of the idx/ptr arrays of each index. The cost of this
  /// function is O(#level).
  TensorStorage::Size getSize() const;

  /// Returns the total size of storage in bytes.
  int numBytes() const;

  /// Returns the index for the given level.  The index content is determined
  /// by the level type, which can be read from the format.
  const TensorStorage::LevelIndex& getLevelIndex(size_t level) const;

  /// Returns the index for the given level.  The index content is determined
  /// by the level type, which can be read from the format.
  TensorStorage::LevelIndex& getLevelIndex(size_t level);

  /// Returns the value array that contains the tensor components.
  const double* getValues() const;

  /// Returns the tensor component value array.
  double* getValues();

  /// True iff the storage has been defined.
  bool defined() const;

private:
  struct Content;
  std::shared_ptr<Content> content;
};

std::ostream& operator<<(std::ostream&, const TensorStorage&);

}}
#endif
