#include "taco/storage/typed_value.h"

using namespace std;

namespace taco {

////////// TypedComponent

const Datatype& TypedComponent::getType() const {
  return dType;
}

size_t TypedComponent::getAsIndex(const ComponentTypeUnion mem) const {
  switch (dType.getKind()) {
    case Datatype::Bool: return (size_t) mem.boolValue;
    case Datatype::UInt8: return (size_t) mem.uint8Value;
    case Datatype::UInt16: return (size_t) mem.uint16Value;
    case Datatype::UInt32: return (size_t) mem.uint32Value;
    case Datatype::UInt64: return (size_t) mem.uint64Value;
    case Datatype::UInt128: return (size_t) mem.uint128Value;
    case Datatype::Int8: return (size_t) mem.int8Value;
    case Datatype::Int16: return (size_t) mem.int16Value;
    case Datatype::Int32: return (size_t) mem.int32Value;
    case Datatype::Int64: return (size_t) mem.int64Value;
    case Datatype::Int128: return (size_t) mem.int128Value;
    case Datatype::Float32: return (size_t) mem.float32Value;
    case Datatype::Float64: return (size_t) mem.float64Value;
    case Datatype::Complex64: taco_ierror; return 0;
    case Datatype::Complex128: taco_ierror; return 0;
    case Datatype::Undefined: taco_ierror; return 0;
  }
  taco_unreachable;
  return 0;
}

void TypedComponent::set(ComponentTypeUnion& mem, const ComponentTypeUnion& value) {
  switch (dType.getKind()) {
    case Datatype::Bool: mem.boolValue = value.boolValue; break;
    case Datatype::UInt8: mem.uint8Value = value.uint8Value; break;
    case Datatype::UInt16: mem.uint16Value = value.uint16Value; break;
    case Datatype::UInt32: mem.uint32Value = value.uint32Value; break;
    case Datatype::UInt64: mem.uint64Value = value.uint64Value; break;
    case Datatype::UInt128: mem.uint128Value = value.uint128Value; break;
    case Datatype::Int8: mem.int8Value = value.int8Value; break;
    case Datatype::Int16: mem.int16Value = value.int16Value; break;
    case Datatype::Int32: mem.int32Value = value.int32Value; break;
    case Datatype::Int64: mem.int64Value = value.int64Value; break;
    case Datatype::Int128: mem.int128Value = value.int128Value; break;
    case Datatype::Float32: mem.float32Value = value.float32Value; break;
    case Datatype::Float64: mem.float64Value = value.float64Value; break;
    case Datatype::Complex64:  mem.complex64Value = value.complex64Value;; break;
    case Datatype::Complex128:  mem.complex128Value = value.complex128Value;; break;
    case Datatype::Undefined: taco_ierror; break;
  }
}

void TypedComponent::setInt(ComponentTypeUnion& mem, const int value) {
  switch (dType.getKind()) {
    case Datatype::Bool: mem.boolValue = value; break;
    case Datatype::UInt8: mem.uint8Value = value; break;
    case Datatype::UInt16: mem.uint16Value = value; break;
    case Datatype::UInt32: mem.uint32Value = value; break;
    case Datatype::UInt64: mem.uint64Value = value; break;
    case Datatype::UInt128: mem.uint128Value = value; break;
    case Datatype::Int8: mem.int8Value = value; break;
    case Datatype::Int16: mem.int16Value = value; break;
    case Datatype::Int32: mem.int32Value = value; break;
    case Datatype::Int64: mem.int64Value = value; break;
    case Datatype::Int128: mem.int128Value = value; break;
    case Datatype::Float32: mem.float32Value = value; break;
    case Datatype::Float64: mem.float64Value = value; break;
    case Datatype::Complex64:  mem.complex64Value = value; break;
    case Datatype::Complex128:  mem.complex128Value = value; break;
    case Datatype::Undefined: taco_ierror; break;
  }
}

void TypedComponent::add(ComponentTypeUnion& result, const ComponentTypeUnion& a, const ComponentTypeUnion& b) const {
  switch (dType.getKind()) {
    case Datatype::Bool: result.boolValue = a.boolValue + b.boolValue; break;
    case Datatype::UInt8: result.uint8Value  = a.uint8Value + b.uint8Value; break;
    case Datatype::UInt16: result.uint16Value  = a.uint16Value + b.uint16Value; break;
    case Datatype::UInt32: result.uint32Value  = a.uint32Value + b.uint32Value; break;
    case Datatype::UInt64: result.uint64Value  = a.uint64Value + b.uint64Value; break;
    case Datatype::UInt128: result.uint128Value  = a.uint128Value +b.uint128Value; break;
    case Datatype::Int8: result.int8Value  = a.int8Value + b.int8Value; break;
    case Datatype::Int16: result.int16Value  = a.int16Value + b.int16Value; break;
    case Datatype::Int32: result.int32Value  = a.int32Value +b.int32Value; break;
    case Datatype::Int64: result.int64Value  = a.int64Value + b.int64Value; break;
    case Datatype::Int128: result.int128Value  = a.int128Value + b.int128Value; break;
    case Datatype::Float32: result.float32Value  = a.float32Value + b.float32Value; break;
    case Datatype::Float64: result.float64Value  = a.float64Value + b.float64Value; break;
    case Datatype::Complex64: result.complex64Value  = a.complex64Value + b.complex64Value; break;
    case Datatype::Complex128: result.complex128Value  = a.complex128Value + b.complex128Value; break;
    case Datatype::Undefined: taco_ierror; break;
  }
}

void TypedComponent::addInt(ComponentTypeUnion& result, const ComponentTypeUnion& a, const int b) const {
  switch (dType.getKind()) {
    case Datatype::Bool: result.boolValue = a.boolValue + (bool) b; break;
    case Datatype::UInt8: result.uint8Value  = a.uint8Value + b; break;
    case Datatype::UInt16: result.uint16Value  = a.uint16Value + b; break;
    case Datatype::UInt32: result.uint32Value  = a.uint32Value + b; break;
    case Datatype::UInt64: result.uint64Value  = a.uint64Value + b; break;
    case Datatype::UInt128: result.uint128Value  = a.uint128Value + b; break;
    case Datatype::Int8: result.int8Value  = a.int8Value + b; break;
    case Datatype::Int16: result.int16Value  = a.int16Value + b; break;
    case Datatype::Int32: result.int32Value  = a.int32Value + b; break;
    case Datatype::Int64: result.int64Value  = a.int64Value + b; break;
    case Datatype::Int128: result.int128Value  = a.int128Value + b; break;
    case Datatype::Float32: result.float32Value  = a.float32Value + b; break;
    case Datatype::Float64: result.float64Value  = a.float64Value + b; break;
    case Datatype::Complex64: result.complex64Value  = a.complex64Value + std::complex<float>(b, 0); break;
    case Datatype::Complex128: result.complex128Value  = a.complex128Value + std::complex<double>(b, 0); break;
    case Datatype::Undefined: taco_ierror; break;
  }
}

void TypedComponent::negate(ComponentTypeUnion& result, const ComponentTypeUnion& a) const {
  switch (dType.getKind()) {
    case Datatype::Bool:
    case Datatype::UInt8:
    case Datatype::UInt16:
    case Datatype::UInt32:
    case Datatype::UInt64:
    case Datatype::UInt128:
      taco_ierror;
      break;
    case Datatype::Int8: result.int8Value  = -a.int8Value; break;
    case Datatype::Int16: result.int16Value  = -a.int16Value; break;
    case Datatype::Int32: result.int32Value  = -a.int32Value; break;
    case Datatype::Int64: result.int64Value  = -a.int64Value; break;
    case Datatype::Int128: result.int128Value  = -a.int128Value; break;
    case Datatype::Float32: result.float32Value  = -a.float32Value; break;
    case Datatype::Float64: result.float64Value  = -a.float64Value; break;
    case Datatype::Complex64: result.complex64Value  = -a.complex64Value; break;
    case Datatype::Complex128: result.complex128Value  = -a.complex128Value; break;
    case Datatype::Undefined: taco_ierror; break;
  }
}

void TypedComponent::multiply(ComponentTypeUnion& result, const ComponentTypeUnion& a, const ComponentTypeUnion& b) const {
  switch (dType.getKind()) {
    case Datatype::Bool: result.boolValue = a.boolValue && b.boolValue; break;
    case Datatype::UInt8: result.uint8Value  = a.uint8Value * b.uint8Value; break;
    case Datatype::UInt16: result.uint16Value  = a.uint16Value * b.uint16Value; break;
    case Datatype::UInt32: result.uint32Value  = a.uint32Value * b.uint32Value; break;
    case Datatype::UInt64: result.uint64Value  = a.uint64Value * b.uint64Value; break;
    case Datatype::UInt128: result.uint128Value  = a.uint128Value *b.uint128Value; break;
    case Datatype::Int8: result.int8Value  = a.int8Value * b.int8Value; break;
    case Datatype::Int16: result.int16Value  = a.int16Value * b.int16Value; break;
    case Datatype::Int32: result.int32Value  = a.int32Value *b.int32Value; break;
    case Datatype::Int64: result.int64Value  = a.int64Value * b.int64Value; break;
    case Datatype::Int128: result.int128Value  = a.int128Value * b.int128Value; break;
    case Datatype::Float32: result.float32Value  = a.float32Value * b.float32Value; break;
    case Datatype::Float64: result.float64Value  = a.float64Value * b.float64Value; break;
    case Datatype::Complex64: result.complex64Value  = a.complex64Value * b.complex64Value; break;
    case Datatype::Complex128: result.complex128Value  = a.complex128Value * b.complex128Value; break;
    case Datatype::Undefined: taco_ierror; break;
  }
}

void TypedComponent::multiplyInt(ComponentTypeUnion& result, const ComponentTypeUnion& a, const int b) const {
  switch (dType.getKind()) {
    case Datatype::Bool: result.boolValue = a.boolValue && (bool) b; break;
    case Datatype::UInt8: result.uint8Value  = a.uint8Value * b; break;
    case Datatype::UInt16: result.uint16Value  = a.uint16Value * b; break;
    case Datatype::UInt32: result.uint32Value  = a.uint32Value * b; break;
    case Datatype::UInt64: result.uint64Value  = a.uint64Value * b; break;
    case Datatype::UInt128: result.uint128Value  = a.uint128Value *b; break;
    case Datatype::Int8: result.int8Value  = a.int8Value * b; break;
    case Datatype::Int16: result.int16Value  = a.int16Value * b; break;
    case Datatype::Int32: result.int32Value  = a.int32Value *b; break;
    case Datatype::Int64: result.int64Value  = a.int64Value * b; break;
    case Datatype::Int128: result.int128Value  = a.int128Value * b; break;
    case Datatype::Float32: result.float32Value  = a.float32Value * b; break;
    case Datatype::Float64: result.float64Value  = a.float64Value * b; break;
    case Datatype::Complex64: result.complex64Value  = a.complex64Value * std::complex<float>(b, 0); break;
    case Datatype::Complex128: result.complex128Value  = a.complex128Value * std::complex<double>(b, 0); break;
    case Datatype::Undefined: taco_ierror; break;
  }
}


////////// TypedComponentVal

TypedComponentVal::TypedComponentVal() {
  dType = Datatype::Undefined;
}

TypedComponentVal::TypedComponentVal(Datatype t) {
  dType = t;
}

TypedComponentVal::TypedComponentVal(TypedComponentRef ref) : val(ref.get()) {
  dType = ref.getType();
}

TypedComponentVal::TypedComponentVal(Datatype t, int constant) {
  dType = t;
  set(constant);
}

ComponentTypeUnion& TypedComponentVal::get() {
  return val;
}

ComponentTypeUnion TypedComponentVal::get() const {
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

TypedComponentVal TypedComponentVal::operator-() const {
  TypedComponentVal result(dType);
  negate(result.get(), val);
  return result;
}

TypedComponentVal TypedComponentVal::operator-(const TypedComponentVal other) const {
  taco_iassert(dType == other.getType());
  if (dType.isUInt()) {
    TypedComponentVal result(dType);
    switch(dType.getKind()) {
      case Datatype::UInt8: result.get().uint8Value  = val.uint8Value - other.get().uint8Value; break;
      case Datatype::UInt16: result.get().uint16Value  = val.uint16Value - other.get().uint16Value; break;
      case Datatype::UInt32: result.get().uint32Value  = val.uint32Value - other.get().uint32Value; break;
      case Datatype::UInt64: result.get().uint64Value  = val.uint64Value - other.get().uint64Value; break;
      case Datatype::UInt128: result.get().uint128Value  = val.uint128Value - other.get().uint128Value; break;
      default:
        taco_ierror;
    }
    return result;
  }
  return (-other) + *this;
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

TypedComponentPtr::TypedComponentPtr(Datatype type, void *ptr) : type(type), ptr(ptr) {
}

void* TypedComponentPtr::get() {
  return ptr;
}

const void* TypedComponentPtr::get() const {
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

ComponentTypeUnion& TypedComponentRef::get() {
  return *ptr;
}

ComponentTypeUnion TypedComponentRef::get() const {
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

TypedComponentVal TypedComponentRef::operator-() const {
  TypedComponentVal result(dType);
  negate(result.get(), *ptr);
  return result;
}

TypedComponentVal TypedComponentRef::operator-(const TypedComponentVal other) const {
  taco_iassert(dType == other.getType());
  if (dType.isUInt()) {
    TypedComponentVal result(dType);
    switch(dType.getKind()) {
      case Datatype::UInt8: result.get().uint8Value  = (*ptr).uint8Value - other.get().uint8Value; break;
      case Datatype::UInt16: result.get().uint16Value  = (*ptr).uint16Value - other.get().uint16Value; break;
      case Datatype::UInt32: result.get().uint32Value  = (*ptr).uint32Value - other.get().uint32Value; break;
      case Datatype::UInt64: result.get().uint64Value  = (*ptr).uint64Value - other.get().uint64Value; break;
      case Datatype::UInt128: result.get().uint128Value  = (*ptr).uint128Value - other.get().uint128Value; break;
      default:
        taco_ierror;
    }
    return result;
  }
  return (-other) + *this;
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
    case Datatype::Bool: return a.get().boolValue > (other.get()).boolValue;
    case Datatype::UInt8: return a.get().uint8Value > (other.get()).uint8Value;
    case Datatype::UInt16: return a.get().uint16Value > (other.get()).uint16Value;
    case Datatype::UInt32: return a.get().uint32Value > (other.get()).uint32Value;
    case Datatype::UInt64: return a.get().uint64Value > (other.get()).uint64Value;
    case Datatype::UInt128: return a.get().uint128Value > (other.get()).uint128Value;
    case Datatype::Int8: return a.get().int8Value > (other.get()).int8Value;
    case Datatype::Int16: return a.get().int16Value > (other.get()).int16Value;
    case Datatype::Int32: return a.get().int32Value > (other.get()).int32Value;
    case Datatype::Int64: return a.get().int64Value > (other.get()).int64Value;
    case Datatype::Int128: return a.get().int128Value > (other.get()).int128Value;
    case Datatype::Float32: return a.get().float32Value > (other.get()).float32Value;
    case Datatype::Float64: return a.get().float64Value > (other.get()).float64Value;
    case Datatype::Complex64: taco_ierror; return false;
    case Datatype::Complex128: taco_ierror; return false;
    case Datatype::Undefined: taco_ierror; return false;
  }
  taco_unreachable;
  return false;
}

bool operator==(const TypedComponentVal& a, const TypedComponentVal &other) {
  taco_iassert(a.getType() == other.getType());
  switch (a.getType().getKind()) {
    case Datatype::Bool: return a.get().boolValue == (other.get()).boolValue;
    case Datatype::UInt8: return a.get().uint8Value == (other.get()).uint8Value;
    case Datatype::UInt16: return a.get().uint16Value == (other.get()).uint16Value;
    case Datatype::UInt32: return a.get().uint32Value == (other.get()).uint32Value;
    case Datatype::UInt64: return a.get().uint64Value == (other.get()).uint64Value;
    case Datatype::UInt128: return a.get().uint128Value == (other.get()).uint128Value;
    case Datatype::Int8: return a.get().int8Value == (other.get()).int8Value;
    case Datatype::Int16: return a.get().int16Value == (other.get()).int16Value;
    case Datatype::Int32: return a.get().int32Value == (other.get()).int32Value;
    case Datatype::Int64: return a.get().int64Value == (other.get()).int64Value;
    case Datatype::Int128: return a.get().int128Value == (other.get()).int128Value;
    case Datatype::Float32: return a.get().float32Value == (other.get()).float32Value;
    case Datatype::Float64: return a.get().float64Value == (other.get()).float64Value;
    case Datatype::Complex64: taco_ierror; return false;
    case Datatype::Complex128: taco_ierror; return false;
    case Datatype::Undefined: taco_ierror; return false;
  }
  taco_unreachable;
  return false;
}

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
    case Datatype::Bool: return a.get().boolValue > other;
    case Datatype::UInt8: return (signed) a.get().uint8Value > other;
    case Datatype::UInt16: return (signed) a.get().uint16Value > other;
    case Datatype::UInt32: return (signed) a.get().uint32Value > other;
    case Datatype::UInt64: return (signed) a.get().uint64Value > other;
    case Datatype::UInt128: return (signed) a.get().uint128Value > other;
    case Datatype::Int8: return a.get().int8Value > other;
    case Datatype::Int16: return a.get().int16Value > other;
    case Datatype::Int32: return a.get().int32Value > other;
    case Datatype::Int64: return a.get().int64Value > other;
    case Datatype::Int128: return a.get().int128Value > other;
    case Datatype::Float32: return a.get().float32Value > other;
    case Datatype::Float64: return a.get().float64Value > other;
    case Datatype::Complex64: taco_ierror; return false;
    case Datatype::Complex128: taco_ierror; return false;
    case Datatype::Undefined: taco_ierror; return false;
  }
  taco_unreachable;
  return false;
}

bool operator==(const TypedComponentVal& a, const int other) {
  switch (a.getType().getKind()) {
    case Datatype::Bool: return a.get().boolValue == other;
    case Datatype::UInt8: return (signed) a.get().uint8Value == other;
    case Datatype::UInt16: return (signed) a.get().uint16Value == other;
    case Datatype::UInt32: return (signed) a.get().uint32Value == other;
    case Datatype::UInt64: return (signed) a.get().uint64Value == other;
    case Datatype::UInt128: return (signed) a.get().uint128Value == other;
    case Datatype::Int8: return a.get().int8Value == other;
    case Datatype::Int16: return a.get().int16Value == other;
    case Datatype::Int32: return a.get().int32Value == other;
    case Datatype::Int64: return a.get().int64Value == other;
    case Datatype::Int128: return a.get().int128Value == other;
    case Datatype::Float32: return a.get().float32Value == other;
    case Datatype::Float64: return a.get().float64Value == other;
    case Datatype::Complex64: taco_ierror; return false;
    case Datatype::Complex128: taco_ierror; return false;
    case Datatype::Undefined: taco_ierror; return false;
  }
  taco_unreachable;
  return false;
}

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
