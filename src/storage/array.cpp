#include "taco/storage/array.h"

#include "taco/util/uncopyable.h"

namespace taco {
namespace storage {

struct Array::Content : util::Uncopyable {
  int* data;
  size_t size;
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

Array::Array() : content(new Content) {
  content->size = 0;
}

Array::Array(int* array, size_t size, Policy policy) : Array() {
  content->size = size;
  content->data = array;
  content->policy = policy;
}

size_t Array::getSize() const {
  return content->size;
}


size_t Array::getElementSize() const {
  return sizeof(int);
}

int Array::operator[](size_t i) const {
  taco_iassert(i < getSize()) << "array index out of bounds";
  return content->data[i];
}

const int* Array::getData() const {
  return content->data;
}

int* Array::getData() {
  return content->data;
}

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
