#include "taco/storage/typed_index.h"

using namespace std;

namespace taco {

////////// TypedIndex

const Datatype& TypedIndex::getType() const {
  return dType;
}

size_t TypedIndex::getAsIndex(const IndexTypeUnion& mem) const {
  switch (dType.getKind()) {
    case Datatype::UInt8: return (size_t) mem.uint8Value;
    case Datatype::UInt16: return (size_t) mem.uint16Value;
    case Datatype::UInt32: return (size_t) mem.uint32Value;
    case Datatype::UInt64: return (size_t) mem.uint64Value;
    case Datatype::Int8: return (size_t) mem.int8Value;
    case Datatype::Int16: return (size_t) mem.int16Value;
    case Datatype::Int32: return (size_t) mem.int32Value;
    case Datatype::Int64: return (size_t) mem.int64Value;
    case Datatype::Bool:
    case Datatype::UInt128:
    case Datatype::Int128:
    case Datatype::Float32:
    case Datatype::Float64:
    case Datatype::Complex64:
    case Datatype::Complex128:
    case Datatype::Undefined: taco_ierror; return 0;
  }
  taco_unreachable;
  return 0;
}

void TypedIndex::set(IndexTypeUnion& mem, const IndexTypeUnion& value) {
  switch (dType.getKind()) {
    case Datatype::UInt8: mem.uint8Value = value.uint8Value; break;
    case Datatype::UInt16: mem.uint16Value = value.uint16Value; break;
    case Datatype::UInt32: mem.uint32Value = value.uint32Value; break;
    case Datatype::UInt64: mem.uint64Value = value.uint64Value; break;
    case Datatype::Int8: mem.int8Value = value.int8Value; break;
    case Datatype::Int16: mem.int16Value = value.int16Value; break;
    case Datatype::Int32: mem.int32Value = value.int32Value; break;
    case Datatype::Int64: mem.int64Value = value.int64Value; break;
    case Datatype::Bool:
    case Datatype::UInt128:
    case Datatype::Int128:
    case Datatype::Float32:
    case Datatype::Float64:
    case Datatype::Complex64:
    case Datatype::Complex128:
    case Datatype::Undefined: taco_ierror;
  }
}

void TypedIndex::setInt(IndexTypeUnion& mem, const int value) {
  switch (dType.getKind()) {
    case Datatype::UInt8: mem.uint8Value = value; break;
    case Datatype::UInt16: mem.uint16Value = value; break;
    case Datatype::UInt32: mem.uint32Value = value; break;
    case Datatype::UInt64: mem.uint64Value = value; break;
    case Datatype::Int8: mem.int8Value = value; break;
    case Datatype::Int16: mem.int16Value = value; break;
    case Datatype::Int32: mem.int32Value = value; break;
    case Datatype::Int64: mem.int64Value = value; break;
    case Datatype::Bool:
    case Datatype::UInt128:
    case Datatype::Int128:
    case Datatype::Float32:
    case Datatype::Float64:
    case Datatype::Complex64:
    case Datatype::Complex128:
    case Datatype::Undefined: taco_ierror;
  }
}

void TypedIndex::add(IndexTypeUnion& result, const IndexTypeUnion& a, const IndexTypeUnion& b) const {
  switch (dType.getKind()) {
    case Datatype::UInt8: result.uint8Value  = a.uint8Value + b.uint8Value; break;
    case Datatype::UInt16: result.uint16Value  = a.uint16Value + b.uint16Value; break;
    case Datatype::UInt32: result.uint32Value  = a.uint32Value + b.uint32Value; break;
    case Datatype::UInt64: result.uint64Value  = a.uint64Value + b.uint64Value; break;
    case Datatype::Int8: result.int8Value  = a.int8Value + b.int8Value; break;
    case Datatype::Int16: result.int16Value  = a.int16Value + b.int16Value; break;
    case Datatype::Int32: result.int32Value  = a.int32Value +b.int32Value; break;
    case Datatype::Int64: result.int64Value  = a.int64Value + b.int64Value; break;
    case Datatype::Bool:
    case Datatype::UInt128:
    case Datatype::Int128:
    case Datatype::Float32:
    case Datatype::Float64:
    case Datatype::Complex64:
    case Datatype::Complex128:
    case Datatype::Undefined: taco_ierror;
  }
}

void TypedIndex::addInt(IndexTypeUnion& result, const IndexTypeUnion& a, const int b) const {
  switch (dType.getKind()) {
    case Datatype::UInt8: result.uint8Value  = a.uint8Value + b; break;
    case Datatype::UInt16: result.uint16Value  = a.uint16Value + b; break;
    case Datatype::UInt32: result.uint32Value  = a.uint32Value + b; break;
    case Datatype::UInt64: result.uint64Value  = a.uint64Value + b; break;
    case Datatype::Int8: result.int8Value  = a.int8Value + b; break;
    case Datatype::Int16: result.int16Value  = a.int16Value + b; break;
    case Datatype::Int32: result.int32Value  = a.int32Value + b; break;
    case Datatype::Int64: result.int64Value  = a.int64Value + b; break;
    case Datatype::Bool:
    case Datatype::UInt128:
    case Datatype::Int128:
    case Datatype::Float32:
    case Datatype::Float64:
    case Datatype::Complex64:
    case Datatype::Complex128:
    case Datatype::Undefined: taco_ierror;
  }
}

void TypedIndex::multiply(IndexTypeUnion& result, const IndexTypeUnion& a, const IndexTypeUnion& b) const {
  switch (dType.getKind()) {
    case Datatype::UInt8: result.uint8Value  = a.uint8Value * b.uint8Value; break;
    case Datatype::UInt16: result.uint16Value  = a.uint16Value * b.uint16Value; break;
    case Datatype::UInt32: result.uint32Value  = a.uint32Value * b.uint32Value; break;
    case Datatype::UInt64: result.uint64Value  = a.uint64Value * b.uint64Value; break;
    case Datatype::Int8: result.int8Value  = a.int8Value * b.int8Value; break;
    case Datatype::Int16: result.int16Value  = a.int16Value * b.int16Value; break;
    case Datatype::Int32: result.int32Value  = a.int32Value *b.int32Value; break;
    case Datatype::Int64: result.int64Value  = a.int64Value * b.int64Value; break;
    case Datatype::Bool:
    case Datatype::UInt128:
    case Datatype::Int128:
    case Datatype::Float32:
    case Datatype::Float64:
    case Datatype::Complex64:
    case Datatype::Complex128:
    case Datatype::Undefined: taco_ierror;
  }
}

void TypedIndex::multiplyInt(IndexTypeUnion& result, const IndexTypeUnion& a, const int b) const {
  switch (dType.getKind()) {
    case Datatype::UInt8: result.uint8Value  = a.uint8Value * b; break;
    case Datatype::UInt16: result.uint16Value  = a.uint16Value * b; break;
    case Datatype::UInt32: result.uint32Value  = a.uint32Value * b; break;
    case Datatype::UInt64: result.uint64Value  = a.uint64Value * b; break;
    case Datatype::Int8: result.int8Value  = a.int8Value * b; break;
    case Datatype::Int16: result.int16Value  = a.int16Value * b; break;
    case Datatype::Int32: result.int32Value  = a.int32Value *b; break;
    case Datatype::Int64: result.int64Value  = a.int64Value * b; break;
    case Datatype::Bool:
    case Datatype::UInt128:
    case Datatype::Int128:
    case Datatype::Float32:
    case Datatype::Float64:
    case Datatype::Complex64:
    case Datatype::Complex128:
    case Datatype::Undefined: taco_ierror;
  }
}

////////// TypedIndexVal

TypedIndexVal::TypedIndexVal() {
  dType = Datatype::Undefined;
}

TypedIndexVal::TypedIndexVal(Datatype t) {
  dType = t;
}

TypedIndexVal::TypedIndexVal(TypedIndexRef ref) : val(ref.get()) {
  dType = ref.getType();
}

IndexTypeUnion& TypedIndexVal::get() {
  return val;
}

IndexTypeUnion TypedIndexVal::get() const {
  return val;
}

size_t TypedIndexVal::getAsIndex() const {
  return TypedIndex::getAsIndex(val);
}

void TypedIndexVal::set(TypedIndexVal value) {
  taco_iassert(dType == value.getType());
  TypedIndex::set(val, value.get());
}

void TypedIndexVal::set(TypedIndexRef value) {
  taco_iassert(dType == value.getType());
  TypedIndex::set(val, value.get());
}

void TypedIndexVal::set(int constant) {
  TypedIndex::setInt(val, constant);
}


void TypedIndexVal::set(TypedComponentVal value) {
  dType = value.getType();
  switch (dType.getKind()) {
    case Datatype::UInt8: val.uint8Value = value.get().uint8Value; break;
    case Datatype::UInt16: val.uint16Value = value.get().uint16Value; break;
    case Datatype::UInt32: val.uint32Value = value.get().uint32Value; break;
    case Datatype::UInt64: val.uint64Value = value.get().uint64Value; break;
    case Datatype::Int8: val.int8Value = value.get().int8Value; break;
    case Datatype::Int16: val.int16Value = value.get().int16Value; break;
    case Datatype::Int32: val.int32Value = value.get().int32Value; break;
    case Datatype::Int64: val.int64Value = value.get().int64Value; break;
    case Datatype::Bool:
    case Datatype::UInt128:
    case Datatype::Int128:
    case Datatype::Float32:
    case Datatype::Float64:
    case Datatype::Complex64:
    case Datatype::Complex128:
    case Datatype::Undefined: taco_ierror; return;  }
}

void TypedIndexVal::set(TypedComponentRef value) {
  dType = value.getType();
  switch (dType.getKind()) {
    case Datatype::UInt8: val.uint8Value = value.get().uint8Value; break;
    case Datatype::UInt16: val.uint16Value = value.get().uint16Value; break;
    case Datatype::UInt32: val.uint32Value = value.get().uint32Value; break;
    case Datatype::UInt64: val.uint64Value = value.get().uint64Value; break;
    case Datatype::Int8: val.int8Value = value.get().int8Value; break;
    case Datatype::Int16: val.int16Value = value.get().int16Value; break;
    case Datatype::Int32: val.int32Value = value.get().int32Value; break;
    case Datatype::Int64: val.int64Value = value.get().int64Value; break;
    case Datatype::Bool:
    case Datatype::UInt128:
    case Datatype::Int128:
    case Datatype::Float32:
    case Datatype::Float64:
    case Datatype::Complex64:
    case Datatype::Complex128:
    case Datatype::Undefined: taco_ierror; return;  }
}

TypedIndexVal TypedIndexVal::operator++() {
  TypedIndexVal copy = *this;
  set(*this + 1);
  return copy;
}

TypedIndexVal TypedIndexVal::operator++(int junk) {
  set(*this + 1);
  return *this;
}

TypedIndexVal TypedIndexVal::operator+(const TypedIndexVal other) const {
  taco_iassert(dType == other.getType());
  TypedIndexVal result(dType);
  add(result.get(), val, other.get());
  return result;
}

TypedIndexVal TypedIndexVal::operator*(const TypedIndexVal other) const {
  taco_iassert(dType == other.getType());
  TypedIndexVal result(dType);
  multiply(result.get(), val, other.get());
  return result;
}

TypedIndexVal TypedIndexVal::operator+(const int other) const {
  TypedIndexVal result(dType);
  addInt(result.get(), val, other);
  return result;
}

TypedIndexVal TypedIndexVal::operator*(const int other) const {
  TypedIndexVal result(dType);
  multiplyInt(result.get(), val, other);
  return result;
}

TypedIndexVal TypedIndexVal::operator=(const int other) {
  set(other);
  return *this;
}

////////// TypedIndexPtr

TypedIndexPtr::TypedIndexPtr() : ptr(nullptr) {}

TypedIndexPtr::TypedIndexPtr (Datatype type, void *ptr) : type(type), ptr(ptr) {
}

void* TypedIndexPtr::get() {
  return ptr;
}

TypedIndexRef TypedIndexPtr::operator*() const {
  return TypedIndexRef(type, ptr);
}

bool TypedIndexPtr::operator> (const TypedIndexPtr &other) const {
  return ptr > other.ptr;
}

bool TypedIndexPtr::operator<= (const TypedIndexPtr &other) const {
  return ptr <= other.ptr;
}

bool TypedIndexPtr::operator< (const TypedIndexPtr &other) const {
  return ptr < other.ptr;
}

bool TypedIndexPtr::operator>= (const TypedIndexPtr &other) const {
  return ptr >= other.ptr;
}

bool TypedIndexPtr::operator== (const TypedIndexPtr &other) const {
  return ptr == other.ptr;
}

bool TypedIndexPtr::operator!= (const TypedIndexPtr &other) const {
  return ptr != other.ptr;
}

TypedIndexPtr TypedIndexPtr::operator+ (int value) const {
  return TypedIndexPtr(type, (char *) ptr + value * type.getNumBytes());
}

TypedIndexPtr TypedIndexPtr::operator++() {
  TypedIndexPtr copy = *this;
  *this = *this + 1;
  return copy;
}

TypedIndexPtr TypedIndexPtr::operator++(int junk) {
  *this = *this + 1;
  return *this;
}

////////// TypedIndexRef

TypedIndexPtr TypedIndexRef::operator&() const {
  return TypedIndexPtr(dType, ptr);
}

IndexTypeUnion& TypedIndexRef::get() {
  return *ptr;
}

IndexTypeUnion TypedIndexRef::get() const {
  return *ptr;
}

size_t TypedIndexRef::getAsIndex() const {
  return TypedIndex::getAsIndex(*ptr);
}

void TypedIndexRef::set(TypedIndexVal value) {
  TypedIndex::set(*ptr, value.get());
}

TypedIndexRef TypedIndexRef::operator=(TypedIndexVal other) {
  set(other);
  return *this;
}

TypedIndexRef TypedIndexRef::operator=(TypedIndexRef other) {
  set(other);
  return *this;
}

TypedIndexRef TypedIndexRef::operator=(const int other) {
  setInt(*ptr, other);
  return *this;
}

TypedIndexRef TypedIndexRef::operator++() {
  TypedIndexRef copy = *this;
  set(*this + 1);
  return copy;
}

TypedIndexRef TypedIndexRef::operator++(int junk) {
  set(*this + 1);
  return *this;
}

TypedIndexVal TypedIndexRef::operator+(const TypedIndexVal other) const {
  TypedIndexVal result(dType);
  add(result.get(), *ptr, other.get());
  return result;
}

TypedIndexVal TypedIndexRef::operator*(const TypedIndexVal other) const {
  TypedIndexVal result(dType);
  multiply(result.get(), *ptr, other.get());
  return result;
}

TypedIndexVal TypedIndexRef::operator+(const int other) const {
  TypedIndexVal result(dType);
  addInt(result.get(), *ptr, other);
  return result;
}

TypedIndexVal TypedIndexRef::operator*(const int other) const {
  TypedIndexVal result(dType);
  multiplyInt(result.get(), *ptr, other);
  return result;
}

////////// Binary Operators

bool operator>(const TypedIndexVal& a, const TypedIndexVal &other) {
  taco_iassert(a.getType() == other.getType());
  switch (a.getType().getKind()) {
    case Datatype::UInt8: return a.get().uint8Value > (other.get()).uint8Value;
    case Datatype::UInt16: return a.get().uint16Value > (other.get()).uint16Value;
    case Datatype::UInt32: return a.get().uint32Value > (other.get()).uint32Value;
    case Datatype::UInt64: return a.get().uint64Value > (other.get()).uint64Value;
    case Datatype::Int8: return a.get().int8Value > (other.get()).int8Value;
    case Datatype::Int16: return a.get().int16Value > (other.get()).int16Value;
    case Datatype::Int32: return a.get().int32Value > (other.get()).int32Value;
    case Datatype::Int64: return a.get().int64Value > (other.get()).int64Value;
    case Datatype::Bool:
    case Datatype::UInt128:
    case Datatype::Int128:
    case Datatype::Float32:
    case Datatype::Float64:
    case Datatype::Complex64:
    case Datatype::Complex128:
    case Datatype::Undefined: taco_ierror; return false;
  }
  taco_unreachable;
  return false;
}

bool operator==(const TypedIndexVal& a, const TypedIndexVal &other) {
  taco_iassert(a.getType() == other.getType());
  switch (a.getType().getKind()) {
    case Datatype::UInt8: return a.get().uint8Value == (other.get()).uint8Value;
    case Datatype::UInt16: return a.get().uint16Value == (other.get()).uint16Value;
    case Datatype::UInt32: return a.get().uint32Value == (other.get()).uint32Value;
    case Datatype::UInt64: return a.get().uint64Value == (other.get()).uint64Value;
    case Datatype::Int8: return a.get().int8Value == (other.get()).int8Value;
    case Datatype::Int16: return a.get().int16Value == (other.get()).int16Value;
    case Datatype::Int32: return a.get().int32Value == (other.get()).int32Value;
    case Datatype::Int64: return a.get().int64Value == (other.get()).int64Value;
    case Datatype::Bool:
    case Datatype::UInt128:
    case Datatype::Int128:
    case Datatype::Float32:
    case Datatype::Float64:
    case Datatype::Complex64:
    case Datatype::Complex128:
    case Datatype::Undefined: taco_ierror; return false;
  }
  taco_unreachable;
  return false;
}

bool operator>=(const TypedIndexVal& a,const TypedIndexVal &other) {
  return (a > other ||a == other);
}

bool operator<(const TypedIndexVal& a, const TypedIndexVal &other) {
  return !(a >= other);
}

bool operator<=(const TypedIndexVal& a, const TypedIndexVal &other) {
  return !(a > other);
}

bool operator!=(const TypedIndexVal& a, const TypedIndexVal &other) {
  return !(a == other);
}

bool operator>(const TypedIndexVal& a, const int other) {
  switch (a.getType().getKind()) {
    case Datatype::UInt8: return (signed) a.get().uint8Value > other;
    case Datatype::UInt16: return (signed) a.get().uint16Value > other;
    case Datatype::UInt32: return (signed) a.get().uint32Value > other;
    case Datatype::UInt64: return (signed) a.get().uint64Value > other;
    case Datatype::Int8: return a.get().int8Value > other;
    case Datatype::Int16: return a.get().int16Value > other;
    case Datatype::Int32: return a.get().int32Value > other;
    case Datatype::Int64: return a.get().int64Value > other;
    case Datatype::Bool:
    case Datatype::UInt128:
    case Datatype::Int128:
    case Datatype::Float32:
    case Datatype::Float64:
    case Datatype::Complex64:
    case Datatype::Complex128:
    case Datatype::Undefined: taco_ierror; return false;
  }
  taco_unreachable;
  return false;
}

bool operator==(const TypedIndexVal& a, const int other) {
  switch (a.getType().getKind()) {
    case Datatype::UInt8: return (signed) a.get().uint8Value == other;
    case Datatype::UInt16: return (signed) a.get().uint16Value == other;
    case Datatype::UInt32: return (signed) a.get().uint32Value == other;
    case Datatype::UInt64: return (signed) a.get().uint64Value == other;
    case Datatype::Int8: return a.get().int8Value == other;
    case Datatype::Int16: return a.get().int16Value == other;
    case Datatype::Int32: return a.get().int32Value == other;
    case Datatype::Int64: return a.get().int64Value == other;
    case Datatype::Bool:
    case Datatype::UInt128:
    case Datatype::Int128:
    case Datatype::Float32:
    case Datatype::Float64:
    case Datatype::Complex64:
    case Datatype::Complex128:
    case Datatype::Undefined: taco_ierror; return false;
  }
  taco_unreachable;
  return false;
}

bool operator>=(const TypedIndexVal& a,const int other) {
  return (a > other ||a == other);
}

bool operator<(const TypedIndexVal& a, const int other) {
  return !(a >= other);
}

bool operator<=(const TypedIndexVal& a, const int other) {
  return !(a > other);
}

bool operator!=(const TypedIndexVal& a, const int other) {
  return !(a == other);
}

}
