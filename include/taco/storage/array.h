#ifndef TACO_STORAGE_ARRAY_H
#define TACO_STORAGE_ARRAY_H

#include <vector>
#include <memory>
#include <ostream>
#include <cstring>

#include "taco/util/collections.h"
#include "taco/error.h"

namespace taco {
namespace storage {

/// An array is a smart pointer to raw memory together with a size and a
/// reclamation policy.
class Array {
public:
  /// The memory reclamation policy of Array objects. UserOwns means the Array
  /// object will not free its data, free means it will reclaim data  with the
  /// C free function and delete means it will reclaim data with delete[].
  enum Policy {UserOwns, Free, Delete};

  /// Construct an empty array.
  Array();

  /// Construct an index array. The ownership policy determines whether the
  /// dimension index will free/delete the memory or leave the responsibility
  /// for freeing to the user.
  Array(int* array, size_t size, Policy policy=UserOwns);

  /// Construct an Array from the values.
  Array(const std::vector<int>& vals)
      : Array(util::copyToArray(vals), vals.size(), Free) {}

  /// Returns the number of array elements
  size_t getSize() const;

  /// Returns the size of each array element
  size_t getElementSize() const;

  /// Returns the ith array element.
  int operator[](size_t i) const;

  /// Returns the array data.
  /// @{
  const int* getData() const;
  int* getData();
  /// @}

private:
  struct Content;
  std::shared_ptr<Content> content;
};

/// Send the array data as text to a stream.
std::ostream& operator<<(std::ostream&, const Array&);

}}
#endif
