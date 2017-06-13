#ifndef TACO_STORAGE_ARRAY_H
#define TACO_STORAGE_ARRAY_H

#include <vector>
#include <memory>
#include <ostream>

#include "taco/error.h"

namespace taco {
namespace storage {

/// An array is a piece of memory together with its size and an reclaim policy.
class Array {
public:
  enum Policy {
    UserOwns,
    Free,
    Delete
  };

private:
  struct Content {
    size_t size;
    int* array;
    Policy policy;

    ~Content() {
      switch (policy) {
        case UserOwns:
          // do nothing
          break;
        case Free:
          free(array);
          break;
        case Delete:
          delete[] array;
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
    content->array = array;
    content->policy = policy;
  }

  /// Construct an Array from the values.
  Array(const std::vector<int>& arrayVals) : content(new Content) {
    content->size = arrayVals.size();
    size_t numbytes = arrayVals.size() * sizeof(int);
    content->array = (int*)malloc(numbytes);
    memcpy(content->array, arrayVals.data(), numbytes);
    content->policy = Free;
  }

  /// Returns the array size
  size_t getSize() const {
    return content->size;
  }

  int operator[](size_t i) const {
    taco_iassert(i < content->size);
    return content->array[i];
  }

private:
  std::shared_ptr<Content> content;
};

std::ostream& operator<<(std::ostream& os, const Array& array) {
  os << "[";
  if (array.getSize() > 0) {
    os << array[0];
  }
  for (size_t i = 1; i < array.getSize(); i++) {
    os << ", " << array[i];
  }
  return os << "]";
}

}}
#endif
