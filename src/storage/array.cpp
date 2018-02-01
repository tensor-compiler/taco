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

TypedValue::TypedValue(DataType type) : type(type), memLocation(malloc(type.getNumBytes())) {
}

  TypedValue::TypedValue(DataType type, void *memLocation) : type(type), memLocation(memLocation) {
  }

  const DataType& TypedValue::getType() const {
    return type;
  }

  void* TypedValue::get() const {
    return memLocation;
  }

  //requires that location has same type
  void TypedValue::set(void *location) {
    memcpy(memLocation, location, type.getNumBytes());

  }

  void TypedValue::set(TypedValue value) {
    taco_iassert(type == value.getType());
    set(value.get());
  }

  void TypedValue::set(int constant) {
    switch (type.getKind()) {
      case DataType::Bool: *((bool *) memLocation) = (bool) constant; break;
      case DataType::UInt8: *((uint8_t *) memLocation) = (uint8_t) constant; break;
      case DataType::UInt16: *((uint16_t *) memLocation) = (uint16_t) constant; break;
      case DataType::UInt32: *((uint32_t *) memLocation) = (uint32_t) constant; break;
      case DataType::UInt64: *((uint64_t *) memLocation) = (uint64_t) constant; break;
      case DataType::UInt128: *((unsigned long long *) memLocation) = (unsigned long long) constant; break;
      case DataType::Int8: *((int8_t *) memLocation) = (int8_t) constant; break;
      case DataType::Int16: *((int16_t *) memLocation) = (int16_t) constant; break;
      case DataType::Int32: *((int32_t *) memLocation) = (int32_t) constant; break;
      case DataType::Int64: *((int64_t *) memLocation) = (int64_t) constant; break;
      case DataType::Int128: *((long long *) memLocation) = (long long) constant; break;
      case DataType::Float32: *((float *) memLocation) = (float) constant; break;
      case DataType::Float64: *((double *) memLocation) = (double) constant; break;
      case DataType::Complex64: taco_ierror; break;
      case DataType::Complex128: taco_ierror; break;
      case DataType::Undefined: taco_ierror; break;
    }
  }

  void TypedValue::freeMemory() {
    free(memLocation);
  }

  bool TypedValue::operator>(TypedValue &other) const {
    taco_iassert(type == other.getType());
    switch (type.getKind()) {
      case DataType::Bool: return *((bool *) memLocation) > *((bool *) other.get());
      case DataType::UInt8: return *((uint8_t *) memLocation) > *((uint8_t *) other.get());
      case DataType::UInt16: return *((uint16_t *) memLocation) > *((uint16_t *) other.get());
      case DataType::UInt32: return *((uint32_t *) memLocation) > *((uint32_t *) other.get());
      case DataType::UInt64: return *((uint64_t *) memLocation) > *((uint64_t *) other.get());
      case DataType::UInt128: return *((unsigned long long *) memLocation) > *((unsigned long long *) other.get());
      case DataType::Int8: return *((int8_t *) memLocation) > *((int8_t *) other.get());
      case DataType::Int16: return *((int16_t *) memLocation) > *((int16_t *) other.get());
      case DataType::Int32: return *((int32_t *) memLocation) > *((int32_t *) other.get());
      case DataType::Int64: return *((int64_t *) memLocation) > *((int64_t *) other.get());
      case DataType::Int128: return *((long long *) memLocation) > *((long long *) other.get());
      case DataType::Float32: return *((float *) memLocation) > *((float *) other.get());
      case DataType::Float64: return *((double *) memLocation) > *((double *) other.get());
      case DataType::Complex64: taco_ierror; return false;
      case DataType::Complex128: taco_ierror; return false;
      case DataType::Undefined: taco_ierror; return false;
    }
  }

  bool TypedValue::operator==(TypedValue &other) const {
    taco_iassert(type == other.getType());
    return memcmp(memLocation, other.get(), type.getNumBytes()) == 0;
  }

  bool TypedValue::operator>=(TypedValue &other) const {
    return (*this > other || *this == other);
  }

  bool TypedValue::operator<(TypedValue &other) const {
    return !(*this >= other);
  }

  bool TypedValue::operator<=(TypedValue &other) const {
    return !(*this > other);
  }

  bool TypedValue::operator!=(TypedValue &other) const {
    return !(*this == other);
  }


}}
