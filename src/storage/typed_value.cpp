#include "taco/storage/typed_value.h"

using namespace std;

namespace taco {
namespace storage {

const DataType& Typed::getType() const {
  return dType;
}

size_t Typed::getAsIndex(const DataTypeUnion mem) const {
  switch (dType.getKind()) {
    case DataType::Bool: return (size_t) mem.boolValue;
    case DataType::UInt8: return (size_t) mem.uint8Value;
    case DataType::UInt16: return (size_t) mem.uint16Value;
    case DataType::UInt32: return (size_t) mem.uint32Value;
    case DataType::UInt64: return (size_t) mem.uint64Value;
    case DataType::UInt128: return (size_t) mem.uint128Value;
    case DataType::Int8: return (size_t) mem.int8Value;
    case DataType::Int16: return (size_t) mem.int16Value;
    case DataType::Int32: return (size_t) mem.int32Value;
    case DataType::Int64: return (size_t) mem.int64Value;
    case DataType::Int128: return (size_t) mem.int128Value;
    case DataType::Float32: return (size_t) mem.float32Value;
    case DataType::Float64: return (size_t) mem.float64Value;
    case DataType::Complex64: taco_ierror; return 0;
    case DataType::Complex128: taco_ierror; return 0;
    case DataType::Undefined: taco_ierror; return 0;
  }
}

void Typed::set(DataTypeUnion& mem, DataTypeUnion value) {
  switch (dType.getKind()) {
    case DataType::Bool: mem.boolValue = value.boolValue; break;
    case DataType::UInt8: mem.uint8Value = value.uint8Value; break;
    case DataType::UInt16: mem.uint16Value = value.uint16Value; break;
    case DataType::UInt32: mem.uint32Value = value.uint32Value; break;
    case DataType::UInt64: mem.uint64Value = value.uint64Value; break;
    case DataType::UInt128: mem.uint128Value = value.uint128Value; break;
    case DataType::Int8: mem.int8Value = value.int8Value; break;
    case DataType::Int16: mem.int16Value = value.int16Value; break;
    case DataType::Int32: mem.int32Value = value.int32Value; break;
    case DataType::Int64: mem.int64Value = value.int64Value; break;
    case DataType::Int128: mem.int128Value = value.int128Value; break;
    case DataType::Float32: mem.float32Value = value.float32Value; break;
    case DataType::Float64: mem.float64Value = value.float64Value; break;
    case DataType::Complex64:  mem.complex64Value = value.complex64Value;; break;
    case DataType::Complex128:  mem.complex128Value = value.complex128Value;; break;
    case DataType::Undefined: taco_ierror; break;
  }
}


// TODO: how to do this differently?
void castToType(DataTypeUnion& result, DataTypeUnion value, DataType valueType, DataType resultType) {
  unsigned long v;
  switch (valueType.getKind()) {
    case DataType::Bool: v = value.boolValue;
    case DataType::UInt8: v = value.uint8Value;
    case DataType::UInt16: v = value.uint16Value;
    case DataType::UInt32: v = value.uint32Value;
    case DataType::UInt64: v = value.uint64Value;
    case DataType::UInt128: v = value.uint128Value;
    case DataType::Int8: v = value.int8Value;
    case DataType::Int16: v = value.int16Value;
    case DataType::Int32: v = value.int32Value;
    case DataType::Int64: v = value.int64Value;
    case DataType::Int128: v = value.int128Value;
    case DataType::Float32: taco_ierror; v= 0;
    case DataType::Float64: taco_ierror; v= 0;
    case DataType::Complex64: taco_ierror; v= 0;
    case DataType::Complex128: taco_ierror; v= 0;
    case DataType::Undefined: taco_ierror; v = 0;
  }

  switch(resultType.getKind()) {
    case DataType::Bool: result.boolValue = v;
    case DataType::UInt8: result.uint8Value = v;
    case DataType::UInt16: result.uint16Value = v;
    case DataType::UInt32: result.uint32Value = v;
    case DataType::UInt64: result.uint64Value = v;
    case DataType::UInt128: result.uint128Value = v;
    case DataType::Int8: result.int8Value = v;
    case DataType::Int16: result.int16Value = v;
    case DataType::Int32: result.int32Value = v;
    case DataType::Int64: result.int64Value = v;
    case DataType::Int128: result.int128Value = v;
    case DataType::Float32: taco_ierror;
    case DataType::Float64: taco_ierror;
    case DataType::Complex64: taco_ierror;
    case DataType::Complex128: taco_ierror;
    case DataType::Undefined: taco_ierror;
  }
}


void Typed::set(DataTypeUnion& mem, DataTypeUnion value, DataType valueType) {
  dType = max_type(dType, valueType);

  if (dType == valueType) {
    set(mem, value);
  }
  else {
    castToType(mem, value, valueType, dType);
  }
}

void Typed::multiply(DataTypeUnion& result, const DataTypeUnion a, const DataTypeUnion b) const {
  result = a;
  switch (dType.getKind()) {
    case DataType::Bool: result.boolValue *= b.boolValue; break;
    case DataType::UInt8: result.uint8Value *= b.uint8Value; break;
    case DataType::UInt16: result.uint16Value *= b.uint16Value; break;
    case DataType::UInt32: result.uint32Value *= b.uint32Value; break;
    case DataType::UInt64: result.uint64Value *= b.uint64Value; break;
    case DataType::UInt128: result.uint128Value *= b.uint128Value; break;
    case DataType::Int8: result.int8Value *= b.int8Value; break;
    case DataType::Int16: result.int16Value *= b.int16Value; break;
    case DataType::Int32: result.int32Value *= b.int32Value; break;
    case DataType::Int64: result.int64Value *= b.int64Value; break;
    case DataType::Int128: result.int128Value *= b.int128Value; break;
    case DataType::Float32: result.float32Value *= b.float32Value; break;
    case DataType::Float64: result.float64Value *= b.float64Value; break;
    case DataType::Complex64: result.complex64Value *= b.complex64Value; break;
    case DataType::Complex128: result.complex128Value *= b.complex128Value; break;
    case DataType::Undefined: taco_ierror; break;
  }
}

void Typed::add(DataTypeUnion& result, const DataTypeUnion a, const DataTypeUnion b) const {
  result = a;
  switch (dType.getKind()) {
    case DataType::Bool: result.boolValue += b.boolValue; break;
    case DataType::UInt8: result.uint8Value += b.uint8Value; break;
    case DataType::UInt16: result.uint16Value += b.uint16Value; break;
    case DataType::UInt32: result.uint32Value += b.uint32Value; break;
    case DataType::UInt64: result.uint64Value += b.uint64Value; break;
    case DataType::UInt128: result.uint128Value += b.uint128Value; break;
    case DataType::Int8: result.int8Value += b.int8Value; break;
    case DataType::Int16: result.int16Value += b.int16Value; break;
    case DataType::Int32: result.int32Value += b.int32Value; break;
    case DataType::Int64: result.int64Value += b.int64Value; break;
    case DataType::Int128: result.int128Value += b.int128Value; break;
    case DataType::Float32: result.float32Value += b.float32Value; break;
    case DataType::Float64: result.float64Value += b.float64Value; break;
    case DataType::Complex64: result.complex64Value += b.complex64Value; break;
    case DataType::Complex128: result.complex128Value += b.complex128Value; break;
    case DataType::Undefined: taco_ierror; break;
  }
}

void TypedValue::set(TypedRef value) {
  Typed::set(val, value.get());
}

TypedValue::TypedValue(TypedRef ref) : val(ref.get()) {
  dType = ref.getType();
}


TypedValue::TypedValue() {
  dType = DataType::Undefined;
}

TypedValue::TypedValue(DataType t) {
  dType = t;
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
