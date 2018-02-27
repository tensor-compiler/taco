#include "taco/storage/typed_value.h"

using namespace std;

namespace taco {
namespace storage {

const DataType& Typed::getType() const {
  return type;
}

size_t Typed::getAsIndex() const {
  switch (type.getKind()) {
    case DataType::Bool: return (size_t) get().boolValue;
    case DataType::UInt8: return (size_t) get().uint8Value;
    case DataType::UInt16: return (size_t) get().uint16Value;
    case DataType::UInt32: return (size_t) get().uint32Value;
    case DataType::UInt64: return (size_t) get().uint64Value;
    case DataType::UInt128: return (size_t) get().uint128Value;
    case DataType::Int8: return (size_t) get().int8Value;
    case DataType::Int16: return (size_t) get().int16Value;
    case DataType::Int32: return (size_t) get().int32Value;
    case DataType::Int64: return (size_t) get().int64Value;
    case DataType::Int128: return (size_t) get().int128Value;
    case DataType::Float32: return (size_t) get().float32Value;
    case DataType::Float64: return (size_t) get().float64Value;
    case DataType::Complex64: taco_ierror; return 0;
    case DataType::Complex128: taco_ierror; return 0;
    case DataType::Undefined: taco_ierror; return 0;
  }
}

void Typed::set(DataTypeUnion value) {
  memcpy(&get(), &value, type.getNumBytes()); // Don't overwrite extra unused bytes
}

void Typed::set(TypedValue value) {
  taco_iassert(type == value.getType());
  set(value.get());
}

void Typed::set(TypedRef value) {
  taco_iassert(type == value.getType());
  set(value.get());
}

Typed& Typed::operator++() {
  Typed& copy = *this;
  set(*this + 1);
  return copy;
}

Typed& Typed::operator++(int junk) {
  set(*this + 1);
  return *this;
}

TypedValue Typed::operator*(const Typed& other) const {
  taco_iassert(getType() == other.getType());
  TypedValue result = other;

  switch (type.getKind()) {
    case DataType::Bool: result.get().boolValue *= get().boolValue; break;
    case DataType::UInt8: result.get().uint8Value *= get().uint8Value; break;
    case DataType::UInt16: result.get().uint16Value *= get().uint16Value; break;
    case DataType::UInt32: result.get().uint32Value *= get().uint32Value; break;
    case DataType::UInt64: result.get().uint64Value *= get().uint64Value; break;
    case DataType::UInt128: result.get().uint128Value *= get().uint128Value; break;
    case DataType::Int8: result.get().int8Value *= get().int8Value; break;
    case DataType::Int16: result.get().int16Value *= get().int16Value; break;
    case DataType::Int32: result.get().int32Value *= get().int32Value; break;
    case DataType::Int64: result.get().int64Value *= get().int64Value; break;
    case DataType::Int128: result.get().int128Value *= get().int128Value; break;
    case DataType::Float32: result.get().float32Value *= get().float32Value; break;
    case DataType::Float64: result.get().float64Value *= get().float64Value; break;
    case DataType::Complex64: result.get().complex64Value *= get().complex64Value; break;
    case DataType::Complex128: result.get().complex128Value *= get().complex128Value; break;
    case DataType::Undefined: taco_ierror; break;
  }
  return result;
}

TypedValue Typed::operator+(const Typed& other) const {
  taco_iassert(getType() == other.getType());
  TypedValue result = other;
  switch (type.getKind()) {
    case DataType::Bool: result.get().boolValue += get().boolValue; break;
    case DataType::UInt8: result.get().uint8Value += get().uint8Value; break;
    case DataType::UInt16: result.get().uint16Value += get().uint16Value; break;
    case DataType::UInt32: result.get().uint32Value += get().uint32Value; break;
    case DataType::UInt64: result.get().uint64Value += get().uint64Value; break;
    case DataType::UInt128: result.get().uint128Value += get().uint128Value; break;
    case DataType::Int8: result.get().int8Value += get().int8Value; break;
    case DataType::Int16: result.get().int16Value += get().int16Value; break;
    case DataType::Int32: result.get().int32Value += get().int32Value; break;
    case DataType::Int64: result.get().int64Value += get().int64Value; break;
    case DataType::Int128: result.get().int128Value += get().int128Value; break;
    case DataType::Float32: result.get().float32Value += get().float32Value; break;
    case DataType::Float64: result.get().float64Value += get().float64Value; break;
    case DataType::Complex64: result.get().complex64Value += get().complex64Value; break;
    case DataType::Complex128: result.get().complex128Value += get().complex128Value; break;
    case DataType::Undefined: taco_ierror; break;
  }

  return result;
}

TypedValue::TypedValue() {
  type = DataType::Undefined;
}

TypedValue::TypedValue(DataType t) {
  type = t;
}

TypedValue::TypedValue(const Typed& val) : val(val.get()) {
    type = val.getType();
}

bool TypedPtr::operator> (const TypedPtr &other) const {
  return ptr > other.ptr;
}

bool TypedPtr::operator<= (const TypedPtr &other) const {
  return ptr <= other.ptr;
}

bool TypedPtr::operator< (const TypedPtr &other) const {
  return ptr < other.ptr;
}

bool TypedPtr::operator>= (const TypedPtr &other) const {
  return ptr >= other.ptr;
}

bool TypedPtr::operator== (const TypedPtr &other) const {
  return ptr == other.ptr;
}

bool TypedPtr::operator!= (const TypedPtr &other) const {
  return ptr != other.ptr;
}

TypedPtr TypedPtr::operator+ (int value) const {
  return TypedPtr(type, (char *) ptr + value * type.getNumBytes());
}

TypedPtr TypedPtr::operator++() {
  TypedPtr copy = *this;
  *this = *this + 1;
  return copy;
}

TypedPtr TypedPtr::operator++(int junk) {
  *this = *this + 1;
  return *this;
}

TypedRef TypedPtr::operator*() const {
  return TypedRef(type, ptr);
}

}}
