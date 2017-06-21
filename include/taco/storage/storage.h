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
/// contains the tensor values and one index per dimension.  The type of each
/// dimension index is determined by the dimension type in the format, and the
/// ordere of the dimension indices is determined by the format dimension order.
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
