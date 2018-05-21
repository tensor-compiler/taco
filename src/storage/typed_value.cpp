#include "taco/storage/typed_value.h"

using namespace std;

namespace taco {
namespace storage {

////////// TypedComponent

const DataType& TypedComponent::getType() const {
  return dType;
}

size_t TypedComponent::getAsIndex(const ValueTypeUnion mem) const {
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

void TypedComponent::set(ValueTypeUnion& mem, const ValueTypeUnion& value) {
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

void TypedComponent::setInt(ValueTypeUnion& mem, const int value) {
  switch (dType.getKind()) {
    case DataType::Bool: mem.boolValue = value; break;
    case DataType::UInt8: mem.uint8Value = value; break;
    case DataType::UInt16: mem.uint16Value = value; break;
    case DataType::UInt32: mem.uint32Value = value; break;
    case DataType::UInt64: mem.uint64Value = value; break;
    case DataType::UInt128: mem.uint128Value = value; break;
    case DataType::Int8: mem.int8Value = value; break;
    case DataType::Int16: mem.int16Value = value; break;
    case DataType::Int32: mem.int32Value = value; break;
    case DataType::Int64: mem.int64Value = value; break;
    case DataType::Int128: mem.int128Value = value; break;
    case DataType::Float32: mem.float32Value = value; break;
    case DataType::Float64: mem.float64Value = value; break;
    case DataType::Complex64:  mem.complex64Value = value; break;
    case DataType::Complex128:  mem.complex128Value = value; break;
    case DataType::Undefined: taco_ierror; break;
  }
}

void TypedComponent::add(ValueTypeUnion& result, const ValueTypeUnion& a, const ValueTypeUnion& b) const {
  switch (dType.getKind()) {
    case DataType::Bool: result.boolValue = a.boolValue + b.boolValue; break;
    case DataType::UInt8: result.uint8Value  = a.uint8Value + b.uint8Value; break;
    case DataType::UInt16: result.uint16Value  = a.uint16Value + b.uint16Value; break;
    case DataType::UInt32: result.uint32Value  = a.uint32Value + b.uint32Value; break;
    case DataType::UInt64: result.uint64Value  = a.uint64Value + b.uint64Value; break;
    case DataType::UInt128: result.uint128Value  = a.uint128Value +b.uint128Value; break;
    case DataType::Int8: result.int8Value  = a.int8Value + b.int8Value; break;
    case DataType::Int16: result.int16Value  = a.int16Value + b.int16Value; break;
    case DataType::Int32: result.int32Value  = a.int32Value +b.int32Value; break;
    case DataType::Int64: result.int64Value  = a.int64Value + b.int64Value; break;
    case DataType::Int128: result.int128Value  = a.int128Value + b.int128Value; break;
    case DataType::Float32: result.float32Value  = a.float32Value + b.float32Value; break;
    case DataType::Float64: result.float64Value  = a.float64Value + b.float64Value; break;
    case DataType::Complex64: result.complex64Value  = a.complex64Value + b.complex64Value; break;
    case DataType::Complex128: result.complex128Value  = a.complex128Value + b.complex128Value; break;
    case DataType::Undefined: taco_ierror; break;
  }
}

void TypedComponent::addInt(ValueTypeUnion& result, const ValueTypeUnion& a, const int b) const {
  switch (dType.getKind()) {
    case DataType::Bool: result.boolValue = a.boolValue + b; break;
    case DataType::UInt8: result.uint8Value  = a.uint8Value + b; break;
    case DataType::UInt16: result.uint16Value  = a.uint16Value + b; break;
    case DataType::UInt32: result.uint32Value  = a.uint32Value + b; break;
    case DataType::UInt64: result.uint64Value  = a.uint64Value + b; break;
    case DataType::UInt128: result.uint128Value  = a.uint128Value + b; break;
    case DataType::Int8: result.int8Value  = a.int8Value + b; break;
    case DataType::Int16: result.int16Value  = a.int16Value + b; break;
    case DataType::Int32: result.int32Value  = a.int32Value + b; break;
    case DataType::Int64: result.int64Value  = a.int64Value + b; break;
    case DataType::Int128: result.int128Value  = a.int128Value + b; break;
    case DataType::Float32: result.float32Value  = a.float32Value + b; break;
    case DataType::Float64: result.float64Value  = a.float64Value + b; break;
    case DataType::Complex64: result.complex64Value  = a.complex64Value + std::complex<float>(b, 0); break;
    case DataType::Complex128: result.complex128Value  = a.complex128Value + std::complex<double>(b, 0); break;
    case DataType::Undefined: taco_ierror; break;
  }
}

void TypedComponent::multiply(ValueTypeUnion& result, const ValueTypeUnion& a, const ValueTypeUnion& b) const {
  switch (dType.getKind()) {
    case DataType::Bool: result.boolValue = a.boolValue * b.boolValue; break;
    case DataType::UInt8: result.uint8Value  = a.uint8Value * b.uint8Value; break;
    case DataType::UInt16: result.uint16Value  = a.uint16Value * b.uint16Value; break;
    case DataType::UInt32: result.uint32Value  = a.uint32Value * b.uint32Value; break;
    case DataType::UInt64: result.uint64Value  = a.uint64Value * b.uint64Value; break;
    case DataType::UInt128: result.uint128Value  = a.uint128Value *b.uint128Value; break;
    case DataType::Int8: result.int8Value  = a.int8Value * b.int8Value; break;
    case DataType::Int16: result.int16Value  = a.int16Value * b.int16Value; break;
    case DataType::Int32: result.int32Value  = a.int32Value *b.int32Value; break;
    case DataType::Int64: result.int64Value  = a.int64Value * b.int64Value; break;
    case DataType::Int128: result.int128Value  = a.int128Value * b.int128Value; break;
    case DataType::Float32: result.float32Value  = a.float32Value * b.float32Value; break;
    case DataType::Float64: result.float64Value  = a.float64Value * b.float64Value; break;
    case DataType::Complex64: result.complex64Value  = a.complex64Value * b.complex64Value; break;
    case DataType::Complex128: result.complex128Value  = a.complex128Value * b.complex128Value; break;
    case DataType::Undefined: taco_ierror; break;
  }
}

void TypedComponent::multiplyInt(ValueTypeUnion& result, const ValueTypeUnion& a, const int b) const {
  switch (dType.getKind()) {
    case DataType::Bool: result.boolValue = a.boolValue * b; break;
    case DataType::UInt8: result.uint8Value  = a.uint8Value * b; break;
    case DataType::UInt16: result.uint16Value  = a.uint16Value * b; break;
    case DataType::UInt32: result.uint32Value  = a.uint32Value * b; break;
    case DataType::UInt64: result.uint64Value  = a.uint64Value * b; break;
    case DataType::UInt128: result.uint128Value  = a.uint128Value *b; break;
    case DataType::Int8: result.int8Value  = a.int8Value * b; break;
    case DataType::Int16: result.int16Value  = a.int16Value * b; break;
    case DataType::Int32: result.int32Value  = a.int32Value *b; break;
    case DataType::Int64: result.int64Value  = a.int64Value * b; break;
    case DataType::Int128: result.int128Value  = a.int128Value * b; break;
    case DataType::Float32: result.float32Value  = a.float32Value * b; break;
    case DataType::Float64: result.float64Value  = a.float64Value * b; break;
    case DataType::Complex64: result.complex64Value  = a.complex64Value * std::complex<float>(b, 0); break;
    case DataType::Complex128: result.complex128Value  = a.complex128Value * std::complex<double>(b, 0); break;
    case DataType::Undefined: taco_ierror; break;
  }
}


////////// TypedComponentVal

TypedComponentVal::TypedComponentVal() {
  dType = DataType::Undefined;
}

TypedComponentVal::TypedComponentVal(DataType t) {
  dType = t;
}

TypedComponentVal::TypedComponentVal(TypedComponentRef ref) : val(ref.get()) {
  dType = ref.getType();
}

TypedComponentVal::TypedComponentVal(DataType t, int constant) {
  dType = t;
  set(constant);
}

ValueTypeUnion& TypedComponentVal::get() {
  return val;
}

ValueTypeUnion TypedComponentVal::get() const {
  return val;
}

size_t TypedComponentVal::getAsIndex() const {
  return TypedComponent::getAsIndex(val);
}

void TypedComponentVal::set(TypedComponentVal value) {
  taco_iassert(dType == value.getType());
  TypedComponent::set(val, value.get());
}

void TypedComponentVal::set(TypedComponentRef value) {
  taco_iassert(dType == value.getType());
  TypedComponent::set(val, value.get());
}

void TypedComponentVal::set(int constant) {
  TypedComponent::setInt(val, constant);
}

TypedComponentVal TypedComponentVal::operator++() {
  TypedComponentVal copy = *this;
  set(*this + 1);
  return copy;
}

TypedComponentVal TypedComponentVal::operator++(int junk) {
  set(*this + 1);
  return *this;
}

TypedComponentVal TypedComponentVal::operator+(const TypedComponentVal other) const {
  taco_iassert(dType == other.getType());
  TypedComponentVal result(dType);
  add(result.get(), val, other.get());
  return result;
}

TypedComponentVal TypedComponentVal::operator*(const TypedComponentVal other) const {
  taco_iassert(dType == other.getType());
  TypedComponentVal result(dType);
  multiply(result.get(), val, other.get());
  return result;
}

TypedComponentVal TypedComponentVal::operator+(const int other) const {
  TypedComponentVal result(dType);
  addInt(result.get(), val, other);
  return result;
}

TypedComponentVal TypedComponentVal::operator*(const int other) const {
  TypedComponentVal result(dType);
  multiplyInt(result.get(), val, other);
  return result;
}

TypedComponentVal TypedComponentVal::operator=(const int other) {
  set(other);
  return *this;
}

////////// TypedComponentPtr

TypedComponentPtr::TypedComponentPtr() : ptr(nullptr) {}

TypedComponentPtr::TypedComponentPtr(DataType type, void *ptr) : type(type), ptr(ptr) {
}

void* TypedComponentPtr::get() {
  return ptr;
}

TypedComponentRef TypedComponentPtr::operator*() const {
  return TypedComponentRef(type, ptr);
}

bool TypedComponentPtr::operator> (const TypedComponentPtr &other) const {
  return ptr > other.ptr;
}

bool TypedComponentPtr::operator<= (const TypedComponentPtr &other) const {
  return ptr <= other.ptr;
}

bool TypedComponentPtr::operator< (const TypedComponentPtr &other) const {
  return ptr < other.ptr;
}

bool TypedComponentPtr::operator>= (const TypedComponentPtr &other) const {
  return ptr >= other.ptr;
}

bool TypedComponentPtr::operator== (const TypedComponentPtr &other) const {
  return ptr == other.ptr;
}

bool TypedComponentPtr::operator!= (const TypedComponentPtr &other) const {
  return ptr != other.ptr;
}

TypedComponentPtr TypedComponentPtr::operator+ (int value) const {
  return TypedComponentPtr(type, (char *) ptr + value * type.getNumBytes());
}

TypedComponentPtr TypedComponentPtr::operator++() {
  TypedComponentPtr copy = *this;
  *this = *this + 1;
  return copy;
}

TypedComponentPtr TypedComponentPtr::operator++(int junk) {
  *this = *this + 1;
  return *this;
}

////////// TypedComponentRef

TypedComponentPtr TypedComponentRef::operator&() const {
  return TypedComponentPtr(dType, ptr);
}

ValueTypeUnion& TypedComponentRef::get() {
  return *ptr;
}

ValueTypeUnion TypedComponentRef::get() const {
  return *ptr;
}

size_t TypedComponentRef::getAsIndex() const {
  return TypedComponent::getAsIndex(*ptr);
}

void TypedComponentRef::set(TypedComponentVal value) {
  taco_iassert(dType == value.getType());
  TypedComponent::set(*ptr, value.get());
}

TypedComponentRef TypedComponentRef::operator=(TypedComponentVal other) {
  set(other);
  return *this;
}

TypedComponentRef TypedComponentRef::operator=(TypedComponentRef other) {
  set(other);
  return *this;
}

TypedComponentRef TypedComponentRef::operator=(const int other) {
  setInt(*ptr, other);
  return *this;
}

TypedComponentRef TypedComponentRef::operator++() {
  TypedComponentRef copy = *this;
  set(*this + 1);
  return copy;
}

TypedComponentRef TypedComponentRef::operator++(int junk) {
  set(*this + 1);
  return *this;
}

TypedComponentVal TypedComponentRef::operator+(const TypedComponentVal other) const {
  TypedComponentVal result(dType);
  add(result.get(), *ptr, other.get());
  return result;
}

TypedComponentVal TypedComponentRef::operator*(const TypedComponentVal other) const {
  TypedComponentVal result(dType);
  multiply(result.get(), *ptr, other.get());
  return result;
}

TypedComponentVal TypedComponentRef::operator+(const int other) const {
  TypedComponentVal result(dType);
  addInt(result.get(), *ptr, other);
  return result;
}

TypedComponentVal TypedComponentRef::operator*(const int other) const {
  TypedComponentVal result(dType);
  multiplyInt(result.get(), *ptr, other);
  return result;
}

////////// Binary Operators

bool operator>(const TypedComponentVal& a, const TypedComponentVal &other) {
  taco_iassert(a.getType() == other.getType());
  switch (a.getType().getKind()) {
    case DataType::Bool: return a.get().boolValue > (other.get()).boolValue;
    case DataType::UInt8: return a.get().uint8Value > (other.get()).uint8Value;
    case DataType::UInt16: return a.get().uint16Value > (other.get()).uint16Value;
    case DataType::UInt32: return a.get().uint32Value > (other.get()).uint32Value;
    case DataType::UInt64: return a.get().uint64Value > (other.get()).uint64Value;
    case DataType::UInt128: return a.get().uint128Value > (other.get()).uint128Value;
    case DataType::Int8: return a.get().int8Value > (other.get()).int8Value;
    case DataType::Int16: return a.get().int16Value > (other.get()).int16Value;
    case DataType::Int32: return a.get().int32Value > (other.get()).int32Value;
    case DataType::Int64: return a.get().int64Value > (other.get()).int64Value;
    case DataType::Int128: return a.get().int128Value > (other.get()).int128Value;
    case DataType::Float32: return a.get().float32Value > (other.get()).float32Value;
    case DataType::Float64: return a.get().float64Value > (other.get()).float64Value;
    case DataType::Complex64: taco_ierror; return false;
    case DataType::Complex128: taco_ierror; return false;
    case DataType::Undefined: taco_ierror; return false;
  }
}

bool operator==(const TypedComponentVal& a, const TypedComponentVal &other) {
  taco_iassert(a.getType() == other.getType());
  switch (a.getType().getKind()) {
    case DataType::Bool: return a.get().boolValue == (other.get()).boolValue;
    case DataType::UInt8: return a.get().uint8Value == (other.get()).uint8Value;
    case DataType::UInt16: return a.get().uint16Value == (other.get()).uint16Value;
    case DataType::UInt32: return a.get().uint32Value == (other.get()).uint32Value;
    case DataType::UInt64: return a.get().uint64Value == (other.get()).uint64Value;
    case DataType::UInt128: return a.get().uint128Value == (other.get()).uint128Value;
    case DataType::Int8: return a.get().int8Value == (other.get()).int8Value;
    case DataType::Int16: return a.get().int16Value == (other.get()).int16Value;
    case DataType::Int32: return a.get().int32Value == (other.get()).int32Value;
    case DataType::Int64: return a.get().int64Value == (other.get()).int64Value;
    case DataType::Int128: return a.get().int128Value == (other.get()).int128Value;
    case DataType::Float32: return a.get().float32Value == (other.get()).float32Value;
    case DataType::Float64: return a.get().float64Value == (other.get()).float64Value;
    case DataType::Complex64: taco_ierror; return false;
    case DataType::Complex128: taco_ierror; return false;
    case DataType::Undefined: taco_ierror; return false;
  }}

bool operator>=(const TypedComponentVal& a,const TypedComponentVal &other) {
  return (a > other ||a == other);
}

bool operator<(const TypedComponentVal& a, const TypedComponentVal &other) {
  return !(a >= other);
}

bool operator<=(const TypedComponentVal& a, const TypedComponentVal &other) {
  return !(a > other);
}

bool operator!=(const TypedComponentVal& a, const TypedComponentVal &other) {
  return !(a == other);
}

  bool operator>(const TypedComponentVal& a, const int other) {
  switch (a.getType().getKind()) {
    case DataType::Bool: return a.get().boolValue > other;
    case DataType::UInt8: return (signed) a.get().uint8Value > other;
    case DataType::UInt16: return (signed) a.get().uint16Value > other;
    case DataType::UInt32: return (signed) a.get().uint32Value > other;
    case DataType::UInt64: return (signed) a.get().uint64Value > other;
    case DataType::UInt128: return (signed) a.get().uint128Value > other;
    case DataType::Int8: return a.get().int8Value > other;
    case DataType::Int16: return a.get().int16Value > other;
    case DataType::Int32: return a.get().int32Value > other;
    case DataType::Int64: return a.get().int64Value > other;
    case DataType::Int128: return a.get().int128Value > other;
    case DataType::Float32: return a.get().float32Value > other;
    case DataType::Float64: return a.get().float64Value > other;
    case DataType::Complex64: taco_ierror; return false;
    case DataType::Complex128: taco_ierror; return false;
    case DataType::Undefined: taco_ierror; return false;
  }
}

bool operator==(const TypedComponentVal& a, const int other) {
  switch (a.getType().getKind()) {
    case DataType::Bool: return a.get().boolValue == other;
    case DataType::UInt8: return (signed) a.get().uint8Value == other;
    case DataType::UInt16: return (signed) a.get().uint16Value == other;
    case DataType::UInt32: return (signed) a.get().uint32Value == other;
    case DataType::UInt64: return (signed) a.get().uint64Value == other;
    case DataType::UInt128: return (signed) a.get().uint128Value == other;
    case DataType::Int8: return a.get().int8Value == other;
    case DataType::Int16: return a.get().int16Value == other;
    case DataType::Int32: return a.get().int32Value == other;
    case DataType::Int64: return a.get().int64Value == other;
    case DataType::Int128: return a.get().int128Value == other;
    case DataType::Float32: return a.get().float32Value == other;
    case DataType::Float64: return a.get().float64Value == other;
    case DataType::Complex64: taco_ierror; return false;
    case DataType::Complex128: taco_ierror; return false;
    case DataType::Undefined: taco_ierror; return false;
  }}

bool operator>=(const TypedComponentVal& a,const int other) {
  return (a > other ||a == other);
}

bool operator<(const TypedComponentVal& a, const int other) {
  return !(a >= other);
}

bool operator<=(const TypedComponentVal& a, const int other) {
  return !(a > other);
}

bool operator!=(const TypedComponentVal& a, const int other) {
  return !(a == other);
}

}
}