#include "taco/storage/typed_index.h"

using namespace std;

namespace taco {
namespace storage {

const DataType& TypedI::getType() const {
  return dType;
}

size_t TypedI::getAsIndex(const IndexTypeUnion mem) const {
  switch (dType.getKind()) {
    case DataType::UInt8: return (size_t) mem.uint8Value;
    case DataType::UInt16: return (size_t) mem.uint16Value;
    case DataType::UInt32: return (size_t) mem.uint32Value;
    case DataType::UInt64: return (size_t) mem.uint64Value;
    case DataType::Int8: return (size_t) mem.int8Value;
    case DataType::Int16: return (size_t) mem.int16Value;
    case DataType::Int32: return (size_t) mem.int32Value;
    case DataType::Int64: return (size_t) mem.int64Value;
    case DataType::Bool:
    case DataType::UInt128:
    case DataType::Int128:
    case DataType::Float32:
    case DataType::Float64:
    case DataType::Complex64:
    case DataType::Complex128:
    case DataType::Undefined: taco_ierror; return 0;
  }
}

void TypedI::set(IndexTypeUnion& mem, IndexTypeUnion value) {
  switch (dType.getKind()) {
    case DataType::UInt8: mem.uint8Value = value.uint8Value; break;
    case DataType::UInt16: mem.uint16Value = value.uint16Value; break;
    case DataType::UInt32: mem.uint32Value = value.uint32Value; break;
    case DataType::UInt64: mem.uint64Value = value.uint64Value; break;
    case DataType::Int8: mem.int8Value = value.int8Value; break;
    case DataType::Int16: mem.int16Value = value.int16Value; break;
    case DataType::Int32: mem.int32Value = value.int32Value; break;
    case DataType::Int64: mem.int64Value = value.int64Value; break;
    case DataType::Bool:
    case DataType::UInt128:
    case DataType::Int128:
    case DataType::Float32:
    case DataType::Float64:
    case DataType::Complex64:
    case DataType::Complex128:
    case DataType::Undefined: taco_ierror; return;  }
}

void TypedI::setInt(IndexTypeUnion& mem, const int value) {
  switch (dType.getKind()) {
    case DataType::UInt8: mem.uint8Value = value; break;
    case DataType::UInt16: mem.uint16Value = value; break;
    case DataType::UInt32: mem.uint32Value = value; break;
    case DataType::UInt64: mem.uint64Value = value; break;
    case DataType::Int8: mem.int8Value = value; break;
    case DataType::Int16: mem.int16Value = value; break;
    case DataType::Int32: mem.int32Value = value; break;
    case DataType::Int64: mem.int64Value = value; break;
    case DataType::Bool:
    case DataType::UInt128:
    case DataType::Int128:
    case DataType::Float32:
    case DataType::Float64:
    case DataType::Complex64:
    case DataType::Complex128:
    case DataType::Undefined: taco_ierror; return;  }
}

void TypedI::add(IndexTypeUnion& result, const IndexTypeUnion a, const IndexTypeUnion b) const {
  switch (dType.getKind()) {
    case DataType::UInt8: result.uint8Value  = a.uint8Value + b.uint8Value; break;
    case DataType::UInt16: result.uint16Value  = a.uint16Value + b.uint16Value; break;
    case DataType::UInt32: result.uint32Value  = a.uint32Value + b.uint32Value; break;
    case DataType::UInt64: result.uint64Value  = a.uint64Value + b.uint64Value; break;
    case DataType::Int8: result.int8Value  = a.int8Value + b.int8Value; break;
    case DataType::Int16: result.int16Value  = a.int16Value + b.int16Value; break;
    case DataType::Int32: result.int32Value  = a.int32Value +b.int32Value; break;
    case DataType::Int64: result.int64Value  = a.int64Value + b.int64Value; break;
    case DataType::Bool:
    case DataType::UInt128:
    case DataType::Int128:
    case DataType::Float32:
    case DataType::Float64:
    case DataType::Complex64:
    case DataType::Complex128:
    case DataType::Undefined: taco_ierror; return;  }
}

void TypedI::addInt(IndexTypeUnion& result, const IndexTypeUnion a, const int b) const {
  switch (dType.getKind()) {
    case DataType::UInt8: result.uint8Value  = a.uint8Value + b; break;
    case DataType::UInt16: result.uint16Value  = a.uint16Value + b; break;
    case DataType::UInt32: result.uint32Value  = a.uint32Value + b; break;
    case DataType::UInt64: result.uint64Value  = a.uint64Value + b; break;
    case DataType::Int8: result.int8Value  = a.int8Value + b; break;
    case DataType::Int16: result.int16Value  = a.int16Value + b; break;
    case DataType::Int32: result.int32Value  = a.int32Value + b; break;
    case DataType::Int64: result.int64Value  = a.int64Value + b; break;
    case DataType::Bool:
    case DataType::UInt128:
    case DataType::Int128:
    case DataType::Float32:
    case DataType::Float64:
    case DataType::Complex64:
    case DataType::Complex128:
    case DataType::Undefined: taco_ierror; return;  }
}

void TypedI::multiply(IndexTypeUnion& result, const IndexTypeUnion a, const IndexTypeUnion b) const {
  switch (dType.getKind()) {
    case DataType::UInt8: result.uint8Value  = a.uint8Value * b.uint8Value; break;
    case DataType::UInt16: result.uint16Value  = a.uint16Value * b.uint16Value; break;
    case DataType::UInt32: result.uint32Value  = a.uint32Value * b.uint32Value; break;
    case DataType::UInt64: result.uint64Value  = a.uint64Value * b.uint64Value; break;
    case DataType::Int8: result.int8Value  = a.int8Value * b.int8Value; break;
    case DataType::Int16: result.int16Value  = a.int16Value * b.int16Value; break;
    case DataType::Int32: result.int32Value  = a.int32Value *b.int32Value; break;
    case DataType::Int64: result.int64Value  = a.int64Value * b.int64Value; break;
    case DataType::Bool:
    case DataType::UInt128:
    case DataType::Int128:
    case DataType::Float32:
    case DataType::Float64:
    case DataType::Complex64:
    case DataType::Complex128:
    case DataType::Undefined: taco_ierror; return;  }
}

void TypedI::multiplyInt(IndexTypeUnion& result, const IndexTypeUnion a, const int b) const {
  switch (dType.getKind()) {
    case DataType::UInt8: result.uint8Value  = a.uint8Value * b; break;
    case DataType::UInt16: result.uint16Value  = a.uint16Value * b; break;
    case DataType::UInt32: result.uint32Value  = a.uint32Value * b; break;
    case DataType::UInt64: result.uint64Value  = a.uint64Value * b; break;
    case DataType::Int8: result.int8Value  = a.int8Value * b; break;
    case DataType::Int16: result.int16Value  = a.int16Value * b; break;
    case DataType::Int32: result.int32Value  = a.int32Value *b; break;
    case DataType::Int64: result.int64Value  = a.int64Value * b; break;
    case DataType::Bool:
    case DataType::UInt128:
    case DataType::Int128:
    case DataType::Float32:
    case DataType::Float64:
    case DataType::Complex64:
    case DataType::Complex128:
    case DataType::Undefined: taco_ierror; return;  }
}

void TypedIndex::set(TypedIndexRef value) {
  TypedI::set(val, value.get());
}

void TypedIndex::set(int constant) {
  TypedI::setInt(val, constant);
}

TypedIndex::TypedIndex(TypedIndexRef ref) : val(ref.get()) {
  dType = ref.getType();
}

void TypedIndex::set(TypedValue value) {
  dType = value.getType();
  switch (dType.getKind()) {
    case DataType::UInt8: val.uint8Value = value.get().uint8Value; break;
    case DataType::UInt16: val.uint16Value = value.get().uint16Value; break;
    case DataType::UInt32: val.uint32Value = value.get().uint32Value; break;
    case DataType::UInt64: val.uint64Value = value.get().uint64Value; break;
    case DataType::Int8: val.int8Value = value.get().int8Value; break;
    case DataType::Int16: val.int16Value = value.get().int16Value; break;
    case DataType::Int32: val.int32Value = value.get().int32Value; break;
    case DataType::Int64: val.int64Value = value.get().int64Value; break;
    case DataType::Bool:
    case DataType::UInt128:
    case DataType::Int128:
    case DataType::Float32:
    case DataType::Float64:
    case DataType::Complex64:
    case DataType::Complex128:
    case DataType::Undefined: taco_ierror; return;  }
}

void TypedIndex::set(TypedRef value) {
  dType = value.getType();
  switch (dType.getKind()) {
    case DataType::UInt8: val.uint8Value = value.get().uint8Value; break;
    case DataType::UInt16: val.uint16Value = value.get().uint16Value; break;
    case DataType::UInt32: val.uint32Value = value.get().uint32Value; break;
    case DataType::UInt64: val.uint64Value = value.get().uint64Value; break;
    case DataType::Int8: val.int8Value = value.get().int8Value; break;
    case DataType::Int16: val.int16Value = value.get().int16Value; break;
    case DataType::Int32: val.int32Value = value.get().int32Value; break;
    case DataType::Int64: val.int64Value = value.get().int64Value; break;
    case DataType::Bool:
    case DataType::UInt128:
    case DataType::Int128:
    case DataType::Float32:
    case DataType::Float64:
    case DataType::Complex64:
    case DataType::Complex128:
    case DataType::Undefined: taco_ierror; return;  }
}

TypedIndex TypedIndex::operator=(const int other) {
  set(other);
  return *this;
}

TypedIndex::TypedIndex() {
  dType = DataType::Undefined;
}

TypedIndex::TypedIndex(DataType t) {
  dType = t;
}

IndexTypeUnion& TypedIndex::get() {
  return val;
}

IndexTypeUnion TypedIndex::get() const {
  return val;
}

const DataType& TypedIndex::getType() const {
  return TypedI::getType();
}

size_t TypedIndex::getAsIndex() const {
  return TypedI::getAsIndex(val);
}

void TypedIndex::set(TypedIndex value) {
  TypedI::set(val, value.get());
}

TypedIndex TypedIndex::operator++() {
  TypedIndex copy = *this;
  set(*this + 1);
  return copy;
}

TypedIndex TypedIndex::operator++(int junk) {
  set(*this + 1);
  return *this;
}

TypedIndex TypedIndex::operator+(const TypedIndex other) const {
  TypedIndex result(dType);
  add(result.get(), val, other.get());
  return result;
}

TypedIndex TypedIndex::operator*(const TypedIndex other) const {
  TypedIndex result(dType);
  multiply(result.get(), val, other.get());
  return result;
}

TypedIndex TypedIndex::operator+(const int other) const {
  TypedIndex result(dType);
  addInt(result.get(), val, other);
  return result;
}

TypedIndex TypedIndex::operator*(const int other) const {
  TypedIndex result(dType);
  multiplyInt(result.get(), val, other);
  return result;
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

TypedIndexRef TypedIndexPtr::operator*() const {
  return TypedIndexRef(type, ptr);
}

void* TypedIndexPtr::get() {
  return ptr;
}

IndexTypeUnion& TypedIndexRef::get() {
  return *ptr;
}

IndexTypeUnion TypedIndexRef::get() const {
  return *ptr;
}

TypedIndexPtr TypedIndexRef::operator&() const {
  return TypedIndexPtr(dType, ptr);
}

void TypedIndexRef::set(TypedIndex value) {
  TypedI::set(*ptr, value.get());
}

TypedIndexRef TypedIndexRef::operator=(TypedIndex other) {
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

TypedIndex TypedIndexRef::operator+(const TypedIndex other) const {
  TypedIndex result(dType);
  add(result.get(), *ptr, other.get());
  return result;
}

TypedIndex TypedIndexRef::operator*(const TypedIndex other) const {
  TypedIndex result(dType);
  multiply(result.get(), *ptr, other.get());
  return result;
}

TypedIndex TypedIndexRef::operator+(const int other) const {
  TypedIndex result(dType);
  addInt(result.get(), *ptr, other);
  return result;
}

TypedIndex TypedIndexRef::operator*(const int other) const {
  TypedIndex result(dType);
  multiplyInt(result.get(), *ptr, other);
  return result;
}

const DataType& TypedIndexRef::getType() const {
  return TypedI::getType();
}

size_t TypedIndexRef::getAsIndex() const {
  return TypedI::getAsIndex(*ptr);
}

bool operator>(const TypedIndex& a, const TypedIndex &other) {
  taco_iassert(a.getType() == other.getType());
  switch (a.getType().getKind()) {
    case DataType::UInt8: return a.get().uint8Value > (other.get()).uint8Value;
    case DataType::UInt16: return a.get().uint16Value > (other.get()).uint16Value;
    case DataType::UInt32: return a.get().uint32Value > (other.get()).uint32Value;
    case DataType::UInt64: return a.get().uint64Value > (other.get()).uint64Value;
    case DataType::Int8: return a.get().int8Value > (other.get()).int8Value;
    case DataType::Int16: return a.get().int16Value > (other.get()).int16Value;
    case DataType::Int32: return a.get().int32Value > (other.get()).int32Value;
    case DataType::Int64: return a.get().int64Value > (other.get()).int64Value;
    case DataType::Bool:
    case DataType::UInt128:
    case DataType::Int128:
    case DataType::Float32:
    case DataType::Float64:
    case DataType::Complex64:
    case DataType::Complex128:
    case DataType::Undefined: taco_ierror; return false;
  }
}

bool operator==(const TypedIndex& a, const TypedIndex &other) {
  taco_iassert(a.getType() == other.getType());
  switch (a.getType().getKind()) {
    case DataType::UInt8: return a.get().uint8Value == (other.get()).uint8Value;
    case DataType::UInt16: return a.get().uint16Value == (other.get()).uint16Value;
    case DataType::UInt32: return a.get().uint32Value == (other.get()).uint32Value;
    case DataType::UInt64: return a.get().uint64Value == (other.get()).uint64Value;
    case DataType::Int8: return a.get().int8Value == (other.get()).int8Value;
    case DataType::Int16: return a.get().int16Value == (other.get()).int16Value;
    case DataType::Int32: return a.get().int32Value == (other.get()).int32Value;
    case DataType::Int64: return a.get().int64Value == (other.get()).int64Value;
    case DataType::Bool:
    case DataType::UInt128:
    case DataType::Int128:
    case DataType::Float32:
    case DataType::Float64:
    case DataType::Complex64:
    case DataType::Complex128:
    case DataType::Undefined: taco_ierror; return false;
  }}

bool operator>=(const TypedIndex& a,const TypedIndex &other) {
  return (a > other ||a == other);
}

bool operator<(const TypedIndex& a, const TypedIndex &other) {
  return !(a >= other);
}

bool operator<=(const TypedIndex& a, const TypedIndex &other) {
  return !(a > other);
}

bool operator!=(const TypedIndex& a, const TypedIndex &other) {
  return !(a == other);
}

  bool operator>(const TypedIndex& a, const int other) {
  switch (a.getType().getKind()) {
    case DataType::UInt8: return (signed) a.get().uint8Value > other;
    case DataType::UInt16: return (signed) a.get().uint16Value > other;
    case DataType::UInt32: return (signed) a.get().uint32Value > other;
    case DataType::UInt64: return (signed) a.get().uint64Value > other;
    case DataType::Int8: return a.get().int8Value > other;
    case DataType::Int16: return a.get().int16Value > other;
    case DataType::Int32: return a.get().int32Value > other;
    case DataType::Int64: return a.get().int64Value > other;
    case DataType::Bool:
    case DataType::UInt128:
    case DataType::Int128:
    case DataType::Float32:
    case DataType::Float64:
    case DataType::Complex64:
    case DataType::Complex128:
    case DataType::Undefined: taco_ierror; return false;
  }
}

bool operator==(const TypedIndex& a, const int other) {
  switch (a.getType().getKind()) {
    case DataType::UInt8: return (signed) a.get().uint8Value == other;
    case DataType::UInt16: return (signed) a.get().uint16Value == other;
    case DataType::UInt32: return (signed) a.get().uint32Value == other;
    case DataType::UInt64: return (signed) a.get().uint64Value == other;
    case DataType::Int8: return a.get().int8Value == other;
    case DataType::Int16: return a.get().int16Value == other;
    case DataType::Int32: return a.get().int32Value == other;
    case DataType::Int64: return a.get().int64Value == other;
    case DataType::Bool:
    case DataType::UInt128:
    case DataType::Int128:
    case DataType::Float32:
    case DataType::Float64:
    case DataType::Complex64:
    case DataType::Complex128:
    case DataType::Undefined: taco_ierror; return false;
  }}

bool operator>=(const TypedIndex& a,const int other) {
  return (a > other ||a == other);
}

bool operator<(const TypedIndex& a, const int other) {
  return !(a >= other);
}

bool operator<=(const TypedIndex& a, const int other) {
  return !(a > other);
}

bool operator!=(const TypedIndex& a, const int other) {
  return !(a == other);
}

}}
