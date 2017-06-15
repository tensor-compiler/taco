#ifndef TACO_STORAGE_ARRAY_H
#define TACO_STORAGE_ARRAY_H

#include <vector>
#include <memory>
#include <ostream>
#include <cstring>

#include "taco/util/collections.h"
#include "taco/util/uncopyable.h"
#include "taco/error.h"

namespace taco {
namespace storage {

/// An array is a piece of memory together with a size and a reclamation policy.
class Array {
public:
  enum Policy {
    UserOwns,
    Free,
    Delete
  };

private:
  struct Content : util::Uncopyable {
    size_t size;
    int* data;
    Policy policy;

    ~Content() {
      switch (policy) {
        case UserOwns:
          // do nothing
          break;
        case Free:
          free(data);
          break;
        case Delete:
          delete[] data;
          break;
      }
    }
  };

public:
  /// Construct an index array. The ownership policy determines whether the
  /// dimension index will free/delete the memory or leave the responsibility
  /// for freeing to the user.
  Array(size_t size, int* array, Policy policy=UserOwns) : content(new Content){
    content->size = size;
    content->data = array;
    content->policy = policy;
  }

  /// Construct an Array from the values.
  Array(const std::initializer_list<int>& vals) : content(new Content) {
    content->size = vals.size();
    content->data = util::copyToArray(vals);
    content->policy = Free;
  }

  /// Construct an Array from the values.
  Array(const std::vector<int>& vals) : content(new Content) {
    content->size = vals.size();
    content->data = util::copyToArray(vals);
    content->policy = Free;
  }

  /// Returns the number of array elements
  size_t getSize() const {
    return content->size;
  }

  /// Returns the size of each array element
  size_t getElementSize() const {
    return sizeof(int);
  }

  /// Returns the ith array element.
  int operator[](size_t i) const {
    taco_iassert(i < content->size);
    return content->data[i];
  }

  /// Returns the array data.
  /// @{
  const int* getData() const {
    return content->data;
  }

  int* getData() {
    return content->data;
  }
  /// @}

  friend std::ostream& operator<<(std::ostream& os, const Array& array) {
    os << "[";
    if (array.getSize() > 0) {
      os << array[0];
    }
    for (size_t i = 1; i < array.getSize(); i++) {
      os << ", " << array[i];
    }
    return os << "]";
  }

private:
  std::shared_ptr<Content> content;
};

}}
#endif
