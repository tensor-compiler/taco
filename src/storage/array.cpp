#include "taco/storage/array.h"

#include <cstring>
#include <iostream>

#include "taco/type.h"
#include "taco/error.h"
#include "taco/util/uncopyable.h"
#include "taco/util/strings.h"
#include "taco/cuda.h"

using namespace std;

namespace taco {

struct Array::Content : util::Uncopyable {
  Datatype   type;
  void*  data;
  size_t size;
  Policy policy = Array::UserOwns;

  ~Content() {
    switch (policy) {
      case UserOwns:
        // do nothing
        break;
      case Free:
        if (should_use_CUDA_unified_memory()) {
          cuda_unified_free(data);
        }
        else {
          free(data);
        }
        break;
      case Delete:
        switch (type.getKind()) {
          case Datatype::Bool:
            delete[] ((bool*)data);
            break;
          case Datatype::UInt8:
            delete[] ((uint8_t*)data);
            break;
          case Datatype::UInt16:
            delete[] ((uint16_t*)data);
            break;
          case Datatype::UInt32:
            delete[] ((uint32_t*)data);
            break;
          case Datatype::UInt64:
            delete[] ((uint64_t*)data);
            break;
          case Datatype::UInt128:
            delete[] ((unsigned long long*)data);
            break;
          case Datatype::Int8:
            delete[] ((int8_t*)data);
            break;
          case Datatype::Int16:
            delete[] ((int16_t*)data);
            break;
          case Datatype::Int32:
            delete[] ((int32_t*)data);
            break;
          case Datatype::Int64:
            delete[] ((int64_t*)data);
            break;
          case Datatype::Int128:
            delete[] ((long long*)data);
            break;
          case Datatype::Float32:
            delete[] ((float*)data);
            break;
          case Datatype::Float64:
            delete[] ((double*)data);
            break;
          case Datatype::Complex64:
            delete[] ((std::complex<float>*)data);
            break;
          case Datatype::Complex128:
            delete[] ((std::complex<double>*)data);
            break;
          case Datatype::Undefined:
            taco_ierror;
            break;
        }
        break;
    }
  }
};

Array::Array() : content(new Content) {
}

Array::Array(Datatype type, void* data, size_t size, Policy policy) : Array() {
  content->type = type;
  content->data = data;
  content->size = size;
  content->policy = policy;
}

const Datatype& Array::getType() const {
  return content->type;
}

size_t Array::getSize() const {
  return content->size;
}

const void* Array::getData() const {
  return content->data;
}

void* Array::getData() {
  return content->data;
}

TypedComponentRef Array::get(size_t index) const {
  return TypedComponentRef(content->type, ((char *) content->data) + content->type.getNumBytes()*index);
}

TypedComponentRef Array::operator[] (const int index) const {
  return TypedComponentRef(content->type, ((char *) content->data) + content->type.getNumBytes()*index);
}

void Array::zero() {
  memset(getData(), 0, getSize() * getType().getNumBytes());
}

template<typename T>
void printData(ostream& os, const Array& array) {
  const T* data = static_cast<const T*>(array.getData());
  os << "[";
  if (array.getSize() > 0) {
    os << data[0];
  }
  for (size_t i = 1; i < array.getSize(); i++) {
    os << ", " << data[i];
  }
  os << "]";
}

std::ostream& operator<<(std::ostream& os, const Array& array) {
  Datatype type = array.getType();
  switch (type.getKind()) {
    case Datatype::Bool:
      printData<bool>(os, array);
      break;
    case Datatype::UInt8:
      printData<uint8_t>(os, array);
      break;
    case Datatype::UInt16:
      printData<uint16_t>(os, array);
      break;
    case Datatype::UInt32:
      printData<uint32_t>(os, array);
      break;
    case Datatype::UInt64:
      printData<uint64_t>(os, array);
      break;
    case Datatype::UInt128:
      printData<unsigned long long>(os, array);
      break;
    case Datatype::Int8:
      printData<int8_t>(os, array);
      break;
    case Datatype::Int16:
      printData<int16_t>(os, array);
      break;
    case Datatype::Int32:
      printData<int32_t>(os, array);
      break;
    case Datatype::Int64:
      printData<int64_t>(os, array);
      break;
    case Datatype::Int128:
      printData<long long>(os, array);
      break;
    case Datatype::Float32:
      printData<float>(os, array);
      break;
    case Datatype::Float64:
      printData<double>(os, array);
      break;
    case Datatype::Complex64:
      printData<std::complex<float>>(os, array);
      break;
    case Datatype::Complex128:
      printData<std::complex<double>>(os, array);
      break;
    case Datatype::Undefined:
      os << "[]";
      break;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, Array::Policy policy) {
  switch (policy) {
    case Array::UserOwns:
      os << "user";
      break;
    case Array::Free:
      os << "free";
      break;
    case Array::Delete:
      os << "delete";
      break;
  }
  return os;
}

Array makeArray(Datatype type, size_t size) {
  if (should_use_CUDA_unified_memory()) {
    return Array(type, cuda_unified_alloc(size * type.getNumBytes()), size, Array::Free);
  }
  else {
    return Array(type, malloc(size * type.getNumBytes()), size, Array::Free);
  }
}

}
