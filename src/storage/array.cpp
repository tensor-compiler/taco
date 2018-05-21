#include "taco/storage/array.h"

#include <cstring>
#include <iostream>

#include "taco/type.h"
#include "taco/error.h"
#include "taco/util/uncopyable.h"
#include "taco/util/strings.h"

using namespace std;

namespace taco {
namespace storage {

struct Array::Content : util::Uncopyable {
  DataType   type;
  void*  data;
  size_t size;
  Policy policy = Array::UserOwns;

  ~Content() {
    switch (policy) {
      case UserOwns:
        // do nothing
        break;
      case Free:
        free(data);
        break;
      case Delete:
        switch (type.getKind()) {
          case DataType::Bool:
            delete[] ((bool*)data);
            break;
          case DataType::UInt8:
            delete[] ((uint8_t*)data);
            break;
          case DataType::UInt16:
            delete[] ((uint16_t*)data);
            break;
          case DataType::UInt32:
            delete[] ((uint32_t*)data);
            break;
          case DataType::UInt64:
            delete[] ((uint64_t*)data);
            break;
          case DataType::UInt128:
            delete[] ((unsigned long long*)data);
            break;
          case DataType::Int8:
            delete[] ((int8_t*)data);
            break;
          case DataType::Int16:
            delete[] ((int16_t*)data);
            break;
          case DataType::Int32:
            delete[] ((int32_t*)data);
            break;
          case DataType::Int64:
            delete[] ((int64_t*)data);
            break;
          case DataType::Int128:
            delete[] ((long long*)data);
            break;
          case DataType::Float32:
            delete[] ((float*)data);
            break;
          case DataType::Float64:
            delete[] ((double*)data);
            break;
          case DataType::Complex64:
            delete[] ((std::complex<float>*)data);
            break;
          case DataType::Complex128:
            delete[] ((std::complex<double>*)data);
            break;
          case DataType::Undefined:
            taco_ierror;
            break;
        }
        break;
    }
  }
};

Array::Array() : content(new Content) {
}

Array::Array(DataType type, void* data, size_t size, Policy policy) : Array() {
  content->type = type;
  content->data = data;
  content->size = size;
  content->policy = policy;
}

const DataType& Array::getType() const {
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

TypedComponentRef Array::get(int index) const {
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
  DataType type = array.getType();
  switch (type.getKind()) {
    case DataType::Bool:
      printData<bool>(os, array);
      break;
    case DataType::UInt8:
      printData<uint8_t>(os, array);
      break;
    case DataType::UInt16:
      printData<uint16_t>(os, array);
      break;
    case DataType::UInt32:
      printData<uint32_t>(os, array);
      break;
    case DataType::UInt64:
      printData<uint64_t>(os, array);
      break;
    case DataType::UInt128:
      printData<unsigned long long>(os, array);
      break;
    case DataType::Int8:
      printData<int8_t>(os, array);
      break;
    case DataType::Int16:
      printData<int16_t>(os, array);
      break;
    case DataType::Int32:
      printData<int32_t>(os, array);
      break;
    case DataType::Int64:
      printData<int64_t>(os, array);
      break;
    case DataType::Int128:
      printData<long long>(os, array);
      break;
    case DataType::Float32:
      printData<float>(os, array);
      break;
    case DataType::Float64:
      printData<double>(os, array);
      break;
    case DataType::Complex64:
      printData<std::complex<float>>(os, array);
      break;
    case DataType::Complex128:
      printData<std::complex<double>>(os, array);
      break;
    case DataType::Undefined:
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

}}
