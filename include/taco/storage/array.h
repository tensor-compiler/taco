#ifndef TACO_STORAGE_ARRAY_H
#define TACO_STORAGE_ARRAY_H

#include <memory>
#include <ostream>

namespace taco {
class Type;
namespace storage {

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
  Array(Type type, void* data, size_t size, Policy policy=Free);

  /// Returns the type of the array elements
  const Type& getType() const;

  /// Returns the number of array elements
  size_t getSize() const;

  /// Returns the array data.
  /// @{
  const void* getData() const;
  void* getData();
  /// @}

  /// Zero the array content
  void zero();

private:
  struct Content;
  std::shared_ptr<Content> content;
};

/// Send the array data as text to a stream.
std::ostream& operator<<(std::ostream&, const Array&);
std::ostream& operator<<(std::ostream&, Array::Policy);

}}
#endif
