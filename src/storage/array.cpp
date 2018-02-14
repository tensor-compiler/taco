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

TypedValue Array::get(int index) const {
  return TypedValue(content->type, ((char *) content->data) + content->type.getNumBytes()*index);
}

TypedValue Array::operator[] (const int index) const {
  return TypedValue(content->type, ((char *) content->data) + content->type.getNumBytes()*index);
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

TypedValue::TypedValue() : type(DataType::Undefined), memAllocced(false) {
}

TypedValue::TypedValue(DataType type) : type(type), memLocation(malloc(type.getNumBytes())), memAllocced(true) {
}
  

TypedValue::TypedValue(const TypedValue& other) : type(other.getType()), memLocation(malloc(other.getType().getNumBytes())), memAllocced(true)  {
  taco_iassert(type == other.getType());
  memcpy(memLocation, other.get(), type.getNumBytes());
}

TypedValue::TypedValue(TypedValue&& other) : type(other.getType()), memLocation(other.memLocation), memAllocced(false)  {
  other.memLocation = nullptr;
}

const DataType& TypedValue::getType() const {
  return type;
}

void* TypedValue::get() const {
  return memLocation;
}

size_t TypedValue::getAsIndex() const {
  switch (type.getKind()) {
    case DataType::Bool: return (unsigned long long) (*((bool *) memLocation));
    case DataType::UInt8: return (unsigned long long) (*((uint8_t *) memLocation));
    case DataType::UInt16: return (unsigned long long) (*((uint16_t *) memLocation));
    case DataType::UInt32: return (unsigned long long) (*((uint32_t *) memLocation));
    case DataType::UInt64: return (unsigned long long) (*((uint64_t *) memLocation));
    case DataType::UInt128: return (*((unsigned long long *) memLocation));
    case DataType::Int8: return (unsigned long long) (*((int8_t *) memLocation));
    case DataType::Int16: return (unsigned long long) (*((int16_t *) memLocation));
    case DataType::Int32: return (unsigned long long) (*((int32_t *) memLocation));
    case DataType::Int64: return (unsigned long long) (*((int64_t *) memLocation));
    case DataType::Int128: return (unsigned long long) (*((long long *) memLocation));
    case DataType::Float32: return (unsigned long long) (*((float *) memLocation));
    case DataType::Float64: return (unsigned long long) (*((double *) memLocation));
    case DataType::Complex64: taco_ierror; return 0;
    case DataType::Complex128: taco_ierror; return 0;
    case DataType::Undefined: taco_ierror; return 0;
  }
}

//requires that location has same type
void TypedValue::set(void *location) {
  memcpy(memLocation, location, type.getNumBytes());

}

void TypedValue::set(TypedValue value) {
  taco_iassert(type == value.getType());
  set(value.get());
}

bool TypedValue::operator>(const TypedValue &other) const {
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

bool TypedValue::operator==(const TypedValue &other) const {
  taco_iassert(type == other.getType());
  return memcmp(memLocation, other.get(), type.getNumBytes()) == 0;
}

bool TypedValue::operator>=(const TypedValue &other) const {
  return (*this > other || *this == other);
}

bool TypedValue::operator<(const TypedValue &other) const {
  return !(*this >= other);
}

bool TypedValue::operator<=(const TypedValue &other) const {
  return !(*this > other);
}

bool TypedValue::operator!=(const TypedValue &other) const {
  return !(*this == other);
}

TypedValue TypedValue::operator+(const TypedValue other) const {
  taco_iassert(getType() == other.getType());
  TypedValue result = other;

  switch (type.getKind()) {
    case DataType::Bool: *((bool *) result.get()) += *((bool *) memLocation); break;
    case DataType::UInt8: *((uint8_t *) result.get()) += *((uint8_t *) memLocation); break;
    case DataType::UInt16: *((uint16_t *) result.get()) += *((uint16_t *) memLocation); break;
    case DataType::UInt32: *((uint32_t *) result.get()) += *((uint32_t *) memLocation); break;
    case DataType::UInt64: *((uint64_t *) result.get()) += *((uint64_t *) memLocation); break;
    case DataType::UInt128: *((unsigned long long *) result.get()) += *((unsigned long long *) memLocation); break;
    case DataType::Int8: *((int8_t *) result.get()) += *((int8_t *) memLocation); break;
    case DataType::Int16: *((int16_t *) result.get()) += *((int16_t *) memLocation); break;
    case DataType::Int32: *((int32_t *) result.get()) += *((int32_t *) memLocation); break;
    case DataType::Int64: *((int64_t *) result.get()) += *((int64_t *) memLocation); break;
    case DataType::Int128: *((long long *) result.get()) += *((long long *) memLocation); break;
    case DataType::Float32: *((float *) result.get()) += *((float *) memLocation); break;
    case DataType::Float64: *((double *) result.get()) += *((double *) memLocation); break;
    case DataType::Complex64: *((std::complex<float> *) result.get()) += *((std::complex<float> *) memLocation); break;
    case DataType::Complex128: *((std::complex<double> *) result.get()) += *((std::complex<double> *) memLocation); break;
    case DataType::Undefined: taco_ierror; break;
  }

  return result;
}

TypedValue TypedValue::operator*(const TypedValue other) const {
  taco_iassert(getType() == other.getType());
  TypedValue result = other;

  switch (type.getKind()) {
    case DataType::Bool: *((bool *) result.get()) *= *((bool *) memLocation); break;
    case DataType::UInt8: *((uint8_t *) result.get()) *= *((uint8_t *) memLocation); break;
    case DataType::UInt16: *((uint16_t *) result.get()) *= *((uint16_t *) memLocation); break;
    case DataType::UInt32: *((uint32_t *) result.get()) *= *((uint32_t *) memLocation); break;
    case DataType::UInt64: *((uint64_t *) result.get()) *= *((uint64_t *) memLocation); break;
    case DataType::UInt128: *((unsigned long long *) result.get()) *= *((unsigned long long *) memLocation); break;
    case DataType::Int8: *((int8_t *) result.get()) *= *((int8_t *) memLocation); break;
    case DataType::Int16: *((int16_t *) result.get()) *= *((int16_t *) memLocation); break;
    case DataType::Int32: *((int32_t *) result.get()) *= *((int32_t *) memLocation); break;
    case DataType::Int64: *((int64_t *) result.get()) *= *((int64_t *) memLocation); break;
    case DataType::Int128: *((long long *) result.get()) *= *((long long *) memLocation); break;
    case DataType::Float32: *((float *) result.get()) *= *((float *) memLocation); break;
    case DataType::Float64: *((double *) result.get()) *= *((double *) memLocation); break;
    case DataType::Complex64: *((std::complex<float> *) result.get()) *= *((std::complex<float> *) memLocation); break;
    case DataType::Complex128: *((std::complex<double> *) result.get()) *= *((std::complex<double> *) memLocation); break;
    case DataType::Undefined: taco_ierror; break;
  }

  return result;
}

TypedValue& TypedValue::operator=(const TypedValue& other) {
  if (&other != this) {
    if(memAllocced) {
      cleanupMemory();
      memLocation = malloc(type.getNumBytes());
      type = other.getType();
      memAllocced = true;
    }
    else {
      taco_iassert(type == other.getType());
    }
    set(other);
  }
  return *this;
}

TypedValue& TypedValue::operator=(TypedValue&& other) {
  if (&other != this) {
    if(memAllocced || type == DataType::Undefined) {
      cleanupMemory();
      memLocation = other.get();
      type = other.getType();
    }
    else {
      taco_iassert(type == other.getType());
    }
  }
  return *this;
}

TypedValue TypedValue::operator++() {
  *this = *this + 1;
  return *this;
}

void TypedValue::cleanupMemory() {
  if (memAllocced) {
    free(memLocation);
    memLocation = nullptr;
    memAllocced = false;
  }
}

TypedValue::~TypedValue() {
  cleanupMemory();
}

}}
