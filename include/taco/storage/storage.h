#ifndef TACO_STORAGE_STORAGE_H
#define TACO_STORAGE_STORAGE_H

#include <vector>
#include <memory>

struct taco_tensor_t;

namespace taco {
class Format;
class Type;
class Datatype;
class Index;
class Array;

/// Storage for a tensor object.  Tensor storage consists of a value array that
/// contains the tensor values and one index per mode.  The type of each
/// mode index is determined by the mode type in the format, and the
/// ordering of the mode indices is determined by the format mode ordering.
class TensorStorage {
public:

  /// Construct tensor storage for the given format.
  TensorStorage(Datatype componentType, const std::vector<int>& dimensions,
                Format format);

  /// Returns the tensor storage format.
  const Format& getFormat() const;

  /// Returns the component type of the tensor storage.
  Datatype getComponentType() const;

  /// Returns the order of the tensor storage.
  int getOrder() const;

  /// Returns the dimensions of the tensor storage.
  const std::vector<int>& getDimensions() const;

  /// Get the tensor index, which describes the non-zero values.
  /// @{
  const Index& getIndex() const;
  Index getIndex();
  /// @}

  /// Returns the value array that contains the tensor components.
  const Array& getValues() const;

  /// Returns the tensor component value array.
  Array getValues();

  /// Returns the size of the storage in bytes.
  size_t getSizeInBytes();

  /// Convert to a taco_tensor_t, whose lifetime is the same as the storage.
  operator struct taco_tensor_t*() const;

  /// Set the tensor index, which describes the non-zero values.
  void setIndex(const Index& index);

  /// Set the tensor component value array.
  void setValues(const Array& values);

private:
  struct Content;
  std::shared_ptr<Content> content;
};

/// Compare tensor storage objects.
bool equals(TensorStorage a, TensorStorage b);

/// Print Storage objects to a stream.
std::ostream& operator<<(std::ostream&, const TensorStorage&);

}
#endif
