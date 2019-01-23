#ifndef TACO_STORAGE_ARRAY_H
#define TACO_STORAGE_ARRAY_H

#include <memory>
#include <ostream>
#include <taco/type.h>
#include <taco/storage/typed_value.h>
#include "taco/util/collections.h"

namespace taco {

/// An array is a smart pointer to raw memory together with an element type,
/// a size (number of elements) and a reclamation policy.
class Array {
public:
  /// The memory reclamation policy of Array objects. UserOwns means the Array
  /// object will not free its data, free means it will reclaim data  with the
  /// C free function and delete means it will reclaim data with delete[].
  enum Policy {UserOwns, Free, Delete};

  /// Construct an empty array of undefined elements.
  Array();

  /// Construct an array of elements of the given type.
  Array(Datatype type, void* data, size_t size, Policy policy=Free);

  /// Returns the type of the array elements
  const Datatype& getType() const;

  /// Returns the number of array elements
  size_t getSize() const;

  /// Returns the array data.
  /// @{
  const void* getData() const;
  void* getData();
  /// @}

  /// Gets the value at a given index
  TypedComponentRef get(size_t index) const;
  /// Gets the value at a given index
  TypedComponentRef operator[] (const int index) const;

  /// Zero the array content
  void zero();

private:
  struct Content;
  std::shared_ptr<Content> content;
};

/// Print the array.
std::ostream& operator<<(std::ostream&, const Array&);

/// Print the array policy.
std::ostream& operator<<(std::ostream&, Array::Policy);

/// Construct an index array. The ownership policy determines whether the
/// mode index will free/delete the memory or leave the responsibility for
/// freeing to the user.
template <typename T>
Array makeArray(T* data, size_t size, Array::Policy policy=Array::UserOwns) {
  return Array(type<T>(), data, size, policy);
}

/// Construct an array of elements of the given type.
Array makeArray(Datatype type, size_t size);

/// Construct an Array from the values.
template <typename T>
Array makeArray(const std::vector<T>& values) {
  Array array = makeArray(type<T>(), values.size());
  memcpy(array.getData(), values.data(), values.size() * sizeof(T));
  return array;
}

/// Construct an Array from the values.
template <typename T>
Array makeArray(const std::initializer_list<T>& values) {
  return makeArray(std::vector<T>(values));
}

}
#endif

