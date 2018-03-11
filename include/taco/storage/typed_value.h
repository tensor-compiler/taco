#ifndef TACO_STORAGE_TYPED_VALUE_H
#define TACO_STORAGE_TYPED_VALUE_H

#include <taco/type.h>

namespace taco {
namespace storage {

class TypedValue;
class TypedRef;

// Holds a dynamically typed value
class Typed {
public:
  const DataType& getType() const;
  size_t getAsIndex(const DataTypeUnion mem) const;

  void set(DataTypeUnion& mem, DataTypeUnion value, DataType valueType);
  void set(DataTypeUnion& mem, DataTypeUnion value);

  void add(DataTypeUnion& result, const DataTypeUnion a, const DataTypeUnion b) const;
  void multiply(DataTypeUnion& result, const DataTypeUnion a, const DataTypeUnion b) const;

  TypedValue operator*(const Typed& other) const;
protected:
  DataType dType;
};

// Allocates a union to hold a dynamically typed value
class TypedValue: public Typed {
public:
  TypedValue();
  TypedValue(DataType type);
  TypedValue(TypedRef ref);

  template<typename T>
  TypedValue(DataType t, T constant) {
    dType = t;
    set(constant);
  }

  template<typename T>
  TypedValue(const T& constant) {
    dType = type<T>();
    set(constant);
  }

  template<typename T>
  TypedValue(DataType t, T *ptr) {
    dType = t;
    switch (dType.getKind()) {
      case DataType::Bool: set(*((bool*) ptr)); break;
      case DataType::UInt8: set(*((uint8_t*) ptr)); break;
      case DataType::UInt16: set(*((uint16_t*) ptr)); break;
      case DataType::UInt32: set(*((uint32_t*) ptr)); break;
      case DataType::UInt64: set(*((uint64_t*) ptr)); break;
      case DataType::UInt128: set(*((unsigned long long*) ptr)); break;
      case DataType::Int8: set(*((int8_t*) ptr)); break;
      case DataType::Int16: set(*((int16_t*) ptr)); break;
      case DataType::Int32: set(*((int32_t*) ptr)); break;
      case DataType::Int64: set(*((int64_t*) ptr)); break;
      case DataType::Int128: set(*((long long*) ptr)); break;
      case DataType::Float32: set(*((float*) ptr)); break;
      case DataType::Float64: set(*((double*) ptr)); break;
      case DataType::Complex64: taco_ierror; break;
      case DataType::Complex128: taco_ierror; break;
      case DataType::Undefined: taco_ierror; break;
    }
  }

  DataTypeUnion& get() {
    return val;
  }

  DataTypeUnion get() const {
    return val;
  }

  const DataType& getType() const {
    return Typed::getType();
  }

  size_t getAsIndex() const {
    return Typed::getAsIndex(val);
  }

  void set(TypedValue value) {
    Typed::set(val, value.get());
  }

  void set(TypedRef value);

  //Casts constant to type
  template<typename T>
  void set(T constant) {
    Typed::set(val, *((DataTypeUnion *) &constant), type<T>());
  }

  TypedValue operator++() {
    TypedValue copy = *this;
    set(*this + 1);
    return copy;
  }

  TypedValue operator++(int junk) {
    set(*this + 1);
    return *this;
  }

  TypedValue operator+(const TypedValue other) const {
    TypedValue result(dType);
    add(result.get(), val, other.get());
    return result;
  }

  TypedValue operator*(const TypedValue other) const {
    TypedValue result(dType);
    multiply(result.get(), val, other.get());
    return result;
  }

private:
  DataTypeUnion val;
};


// dereferences to typedref
class TypedPtr {
public:
  TypedPtr() : ptr(nullptr) {}

  TypedPtr (DataType type, void *ptr) : type(type), ptr(ptr) {
  }

  void* get();

  TypedRef operator*() const;
  
  bool operator> (const TypedPtr &other) const;
  bool operator<= (const TypedPtr &other) const;

  bool operator< (const TypedPtr &other) const;
  bool operator>= (const TypedPtr &other) const;

  bool operator== (const TypedPtr &other) const;
  bool operator!= (const TypedPtr &other) const;

  TypedPtr operator+(int value) const;
  TypedPtr operator++();
  TypedPtr operator++(int junk);

private:
  DataType type;
  void *ptr;
};

class TypedRef: public Typed{
public:
  template<typename T>
  TypedRef(DataType t, T *ptr) : ptr(reinterpret_cast<DataTypeUnion *>(ptr)) {
    dType = t;
  }

  DataTypeUnion& get() {
    return *ptr;
  }

  DataTypeUnion get() const {
    return *ptr;
  }

  TypedPtr operator&() const {
    return TypedPtr(dType, ptr);
  }

  void set(TypedValue value) {
    Typed::set(*ptr, value.get());
  }

  TypedRef operator=(TypedValue other) {
    set(other);
    return *this;
  }

  TypedRef operator=(TypedRef other) {
    set(other);
    return *this;
  }

  TypedRef operator++() {
    TypedRef copy = *this;
    set(*this + 1);
    return copy;
  }

  TypedRef operator++(int junk) {
    set(*this + 1);
    return *this;
  }

  TypedValue operator+(const TypedValue other) const {
    TypedValue result(dType);
    add(result.get(), *ptr, other.get());
    return result;
  }

  TypedValue operator*(const TypedValue other) const {
    TypedValue result(dType);
    multiply(result.get(), *ptr, other.get());
    return result;
  }

private:
  DataTypeUnion *ptr;
};


bool operator>(const TypedValue& a, const TypedValue &other) {
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

bool operator==(const TypedValue& a, const TypedValue &other) {
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

bool operator>=(const TypedValue& a,const TypedValue &other) {
  return (a > other ||a == other);
}

bool operator<(const TypedValue& a, const TypedValue &other) {
  return !(a >= other);
}

bool operator<=(const TypedValue& a, const TypedValue &other) {
  return !(a > other);
}

bool operator!=(const TypedValue& a, const TypedValue &other) {
  return !(a == other);
}
}}
#endif

