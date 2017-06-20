#include "taco/storage/array.h"

#include <iostream>

#include "taco/type.h"
#include "taco/error.h"
#include "taco/util/uncopyable.h"
#include "taco/util/strings.h"

using namespace std;

namespace taco {
namespace storage {

struct Array::Content : util::Uncopyable {
  Type type;
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

Array::Array(Type type, void* data, size_t size, Policy policy) : Array() {
  content->type = type;
  content->data = (int*)data; // TODO: Fixme
  content->size = size;
  content->policy = policy;
}

const Type& Array::getType() const {
  return content->type;
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

const void* Array::getData() const {
  return content->data;
}

void* Array::getData() {
  return content->data;
}

template<typename T>
ostream& printData(ostream& os, const Array& array) {
  const T* data = static_cast<const T*>(array.getData());
  os << "[";
  if (array.getSize() > 0) {
    os << data[0];
  }
  for (size_t i = 1; i < array.getSize(); i++) {
    os << ", " << data[i];
  }
  return os << "]";
}

std::ostream& operator<<(std::ostream& os, const Array& array) {
  return printData<int>(os, array);
}

}}
