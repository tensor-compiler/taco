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

TypedValue::TypedValue() : type(DataType::Undefined) {
}

TypedValue::TypedValue(DataType type) : type(type) {
}

const DataType& TypedValue::getType() const {
  return type;
}

DataTypeUnion TypedValue::get() const {
  return val;
}



size_t TypedValue::getAsIndex() const {
  switch (type.getKind()) {
    case DataType::Bool: return (size_t) val.boolValue;
    case DataType::UInt8: return (size_t) val.uint8Value;
    case DataType::UInt16: return (size_t) val.uint16Value;
    case DataType::UInt32: return (size_t) val.uint32Value;
    case DataType::UInt64: return (size_t)val.uint64Value;
    case DataType::UInt128: return (size_t) val.uint128Value;
    case DataType::Int8: return (size_t) val.int8Value;
    case DataType::Int16: return (size_t) val.int16Value;
    case DataType::Int32: return (size_t) val.int32Value;
    case DataType::Int64: return (size_t) val.int64Value;
    case DataType::Int128: return (size_t) val.int128Value;
    case DataType::Float32: return (size_t) val.float32Value;
    case DataType::Float64: return (size_t) val.float64Value;
    case DataType::Complex64: taco_ierror; return 0;
    case DataType::Complex128: taco_ierror; return 0;
    case DataType::Undefined: taco_ierror; return 0;
  }
}

//requires that location has same type
void TypedValue::set(DataTypeUnion value) {
  val = value;
}

void TypedValue::set(TypedValue value) {
  taco_iassert(type == value.getType());
  set(value.get());
}

bool TypedValue::operator>(const TypedValue &other) const {
  taco_iassert(type == other.getType());
  switch (type.getKind()) {
    case DataType::Bool: return val.boolValue > (other.get()).boolValue;
    case DataType::UInt8: return val.uint8Value > (other.get()).uint8Value;
    case DataType::UInt16: return val.uint16Value > (other.get()).uint16Value;
    case DataType::UInt32: return val.uint32Value > (other.get()).uint32Value;
    case DataType::UInt64: return val.uint64Value > (other.get()).uint64Value;
    case DataType::UInt128: return val.uint128Value > (other.get()).uint128Value;
    case DataType::Int8: return val.int8Value > (other.get()).int8Value;
    case DataType::Int16: return val.int16Value > (other.get()).int16Value;
    case DataType::Int32: return val.int32Value > (other.get()).int32Value;
    case DataType::Int64: return val.int64Value > (other.get()).int64Value;
    case DataType::Int128: return val.int128Value > (other.get()).int128Value;
    case DataType::Float32: return val.float32Value > (other.get()).float32Value;
    case DataType::Float64: return val.float64Value > (other.get()).float64Value;
    case DataType::Complex64: taco_ierror; return false;
    case DataType::Complex128: taco_ierror; return false;
    case DataType::Undefined: taco_ierror; return false;
  }
}

bool TypedValue::operator==(const TypedValue &other) const {
  taco_iassert(type == other.getType());
  switch (type.getKind()) {
    case DataType::Bool: return val.boolValue == (other.get()).boolValue;
    case DataType::UInt8: return val.uint8Value == (other.get()).uint8Value;
    case DataType::UInt16: return val.uint16Value == (other.get()).uint16Value;
    case DataType::UInt32: return val.uint32Value == (other.get()).uint32Value;
    case DataType::UInt64: return val.uint64Value == (other.get()).uint64Value;
    case DataType::UInt128: return val.uint128Value == (other.get()).uint128Value;
    case DataType::Int8: return val.int8Value == (other.get()).int8Value;
    case DataType::Int16: return val.int16Value == (other.get()).int16Value;
    case DataType::Int32: return val.int32Value == (other.get()).int32Value;
    case DataType::Int64: return val.int64Value == (other.get()).int64Value;
    case DataType::Int128: return val.int128Value == (other.get()).int128Value;
    case DataType::Float32: return val.float32Value == (other.get()).float32Value;
    case DataType::Float64: return val.float64Value == (other.get()).float64Value;
    case DataType::Complex64: taco_ierror; return false;
    case DataType::Complex128: taco_ierror; return false;
    case DataType::Undefined: taco_ierror; return false;
  }}

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
    case DataType::Bool: result.val.boolValue += val.boolValue; break;
    case DataType::UInt8: result.val.uint8Value += val.uint8Value; break;
    case DataType::UInt16: result.val.uint16Value += val.uint16Value; break;
    case DataType::UInt32: result.val.uint32Value += val.uint32Value; break;
    case DataType::UInt64: result.val.uint64Value += val.uint64Value; break;
    case DataType::UInt128: result.val.uint128Value += val.uint128Value; break;
    case DataType::Int8: result.val.int8Value += val.int8Value; break;
    case DataType::Int16: result.val.int16Value += val.int16Value; break;
    case DataType::Int32: result.val.int32Value += val.int32Value; break;
    case DataType::Int64: result.val.int64Value += val.int64Value; break;
    case DataType::Int128: result.val.int128Value += val.int128Value; break;
    case DataType::Float32: result.val.float32Value += val.float32Value; break;
    case DataType::Float64: result.val.float64Value += val.float64Value; break;
    case DataType::Complex64: result.val.complex64Value += val.complex64Value; break;
    case DataType::Complex128: result.val.complex128Value += val.complex128Value; break;
    case DataType::Undefined: taco_ierror; break;
  }

  return result;
}

TypedValue TypedValue::operator++() {
  TypedValue copy = *this;
  *this = *this + 1;
  return copy;
}

TypedValue TypedValue::operator++(int junk) {
  *this = *this + 1;
  return *this;
}

TypedValue TypedValue::operator*(const TypedValue other) const {
  taco_iassert(getType() == other.getType());
  TypedValue result = other;

  switch (type.getKind()) {
    case DataType::Bool: result.val.boolValue *= val.boolValue; break;
    case DataType::UInt8: result.val.uint8Value *= val.uint8Value; break;
    case DataType::UInt16: result.val.uint16Value *= val.uint16Value; break;
    case DataType::UInt32: result.val.uint32Value *= val.uint32Value; break;
    case DataType::UInt64: result.val.uint64Value *= val.uint64Value; break;
    case DataType::UInt128: result.val.uint128Value *= val.uint128Value; break;
    case DataType::Int8: result.val.int8Value *= val.int8Value; break;
    case DataType::Int16: result.val.int16Value *= val.int16Value; break;
    case DataType::Int32: result.val.int32Value *= val.int32Value; break;
    case DataType::Int64: result.val.int64Value *= val.int64Value; break;
    case DataType::Int128: result.val.int128Value *= val.int128Value; break;
    case DataType::Float32: result.val.float32Value *= val.float32Value; break;
    case DataType::Float64: result.val.float64Value *= val.float64Value; break;
    case DataType::Complex64: result.val.complex64Value *= val.complex64Value; break;
    case DataType::Complex128: result.val.complex128Value *= val.complex128Value; break;
    case DataType::Undefined: taco_ierror; break;
  }
  return result;
}

void* TypedRef::get() {
  return ptr;
}

bool TypedRef::operator> (const TypedRef &other) const {
  return ptr > other.ptr;
}

bool TypedRef::operator<= (const TypedRef &other) const {
  return ptr <= other.ptr;
}

bool TypedRef::operator< (const TypedRef &other) const {
  return ptr < other.ptr;
}

bool TypedRef::operator>= (const TypedRef &other) const {
  return ptr >= other.ptr;
}

bool TypedRef::operator== (const TypedRef &other) const {
  return ptr == other.ptr;
}

bool TypedRef::operator!= (const TypedRef &other) const {
  return ptr != other.ptr;
}

TypedRef TypedRef::operator+ (int value) const {
  return TypedRef(type, (char *) ptr + value * type.getNumBytes());
}

TypedRef TypedRef::operator++() {
  TypedRef copy = *this;
  *this = *this + 1;
  return copy;
}

TypedRef TypedRef::operator++(int junk) {
  *this = *this + 1;
  return *this;
}

TypedRef TypedValue::operator&() const {
  return TypedRef(type, (void*) &val);
}

}}
