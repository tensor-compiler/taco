#ifndef TACO_STORAGE_STORAGE_H
#define TACO_STORAGE_STORAGE_H

#include <vector>
#include <memory>

namespace taco {
class Format;
namespace storage {
class Index;
class Array;

/// Storage for a tensor object.  Tensor storage consists of a value array that
/// contains the tensor values and one index per mode.  The type of each
/// mode index is determined by the mode type in the format, and the
/// ordering of the mode indices is determined by the format mode ordering.
class Storage {
public:

  /// Construct an undefined tensor storage.
  Storage();

  /// Construct tensor storage for the given format.
  Storage(const Format& format);

  /// Returns the tensor storage format.
  const Format& getFormat() const;

  /// Set the tensor index, which describes the non-zero values.
  void setIndex(const Index& index);

  /// Get the tensor index, which describes the non-zero values.
  /// @{
  const Index& getIndex() const;
  Index getIndex();
  /// @}

  /// Set the tensor component value array.
  void setValues(const Array& values);

  /// Returns the value array that contains the tensor components.
  const Array& getValues() const;

  /// Returns the tensor component value array.
  Array getValues();

  /// Returns the size of the storage in bytes.
  size_t getSizeInBytes();

private:
  struct Content;
  std::shared_ptr<Content> content;
};

/// Print Storage objects to a stream.
std::ostream& operator<<(std::ostream&, const Storage&);

}}
#endif
