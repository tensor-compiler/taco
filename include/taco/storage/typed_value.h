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
  virtual DataTypeUnion get() const =0;
  virtual DataTypeUnion& get() =0;
  size_t getAsIndex() const;

  void set(TypedValue value);
  void set(TypedRef value);
  void set(DataTypeUnion value);

    //Casts constant to type
  template<typename T>
  void set(T constant) {
    switch (type.getKind()) {
      case DataType::Bool: get().boolValue = (bool) constant; break;
      case DataType::UInt8: get().uint8Value = (uint8_t) constant; break;
      case DataType::UInt16: get().uint16Value = (uint16_t) constant; break;
      case DataType::UInt32: get().uint32Value = (uint32_t) constant; break;
      case DataType::UInt64: get().uint64Value = (uint64_t) constant; break;
      case DataType::UInt128: get().uint128Value = (unsigned long long) constant; break;
      case DataType::Int8: get().int8Value = (int8_t) constant; break;
      case DataType::Int16: get().int16Value = (int16_t) constant; break;
      case DataType::Int32: get().int32Value = (int32_t) constant; break;
      case DataType::Int64: get().int64Value = (int64_t) constant; break;
      case DataType::Int128: get().int128Value = (long long) constant; break;
      case DataType::Float32: get().float32Value = (float) constant; break;
      case DataType::Float64: get().float64Value = (double) constant; break;
      case DataType::Complex64: taco_ierror; break; //explicit specialization
      case DataType::Complex128: taco_ierror; break; //explicit specialization
      case DataType::Undefined: taco_ierror; break;
    }
  }

  Typed& operator++();
  Typed& operator++(int junk);
  TypedValue operator+(const Typed& other) const;
  TypedValue operator*(const Typed& other) const;

  virtual Typed& operator=(int other) =0;

protected:
  DataType type;
};
/*
template<>
void Typed::set(std::complex<float> constant) {
  get().complex64Value = (std::complex<float>) constant;
}

template<>
void Typed::set(std::complex<double> constant) {
  get().complex128Value = (std::complex<double>) constant;
}*/

// Allocates a union to hold a dynamically typed value
class TypedValue: public Typed {
public:
  TypedValue();
  TypedValue(DataType type);
  TypedValue(const Typed& val);

  template<typename T>
  TypedValue(DataType t, T constant) {
    type = t;
    set(constant);
  }

  template<typename T>
  TypedValue(DataType t, T *ptr) {
    type = t;
    switch (type.getKind()) {
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

  Typed& operator=(int other) {
    set(other);
    return *this;
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
    type = t;
  }

  DataTypeUnion& get() {
    return *ptr;
  }

  DataTypeUnion get() const {
    return *ptr;
  }

  TypedPtr operator&() const {
    return TypedPtr(type, ptr);
  }

  Typed& operator=(int other) {
    set(other);
    return *this;
  }

  TypedRef operator=(TypedValue other) {
    set(other);
    return *this;
  }

  TypedRef operator=(TypedRef other) {
    set(other);
    return *this;
  }
  
private:
  DataTypeUnion *ptr;
};


template<class T>
inline bool operator> (const Typed& a, const T& b) {
  TypedValue test(a.getType());
  test.set(b);
  return (Typed&) a > (Typed&) test;
}

template<class T>
inline bool operator<= (const Typed& a, const T& b) {
  TypedValue test(a.getType());
  test.set(b);
  return (Typed&) a <= (Typed&) test;
}

template<class T>
inline bool operator< (const Typed& a, const T& b) {
  TypedValue test(a.getType());
  test.set(b);
  return (Typed&) a < (Typed&) test;
}

template<class T>
inline bool operator>= (const Typed& a, const T& b) {
  TypedValue test(a.getType());
  test.set(b);
  return (Typed&) a >= (Typed&) test;
}

template<class T>
inline bool operator!= (const Typed& a, const T& b) {
  TypedValue test(a.getType());
  test.set(b);
  return (Typed&) a != (Typed&) test;
}

template<class T>
inline bool operator== (const Typed& a, const T& b) {
  TypedValue test(a.getType());
  test.set(b);
  return (Typed&) a == (Typed&) test;
}

template<class T>
TypedValue operator+ (const Typed& a, const T& b) {
  TypedValue test(a.getType());
  test.set(b);
  return (Typed &) a + (Typed &) test;
}

template<class T>
TypedValue operator* (const Typed& a, const T& b) {
  TypedValue test(a.getType());
  test.set(b);
  return (Typed &) a * (Typed &) test;
}


template<>
inline bool operator>(const Typed& a, const Typed &other) {
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

template<>
inline bool operator==(const Typed& a, const Typed &other) {
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

template<>
inline bool operator>=(const Typed& a,const Typed &other) {
  return (a > other ||a == other);
}

template<>
inline bool operator<(const Typed& a, const Typed &other) {
  return !(a >= other);
}

template<>
inline bool operator<=(const Typed& a, const Typed &other) {
  return !(a > other);
}

template<>
inline bool operator!=(const Typed& a, const Typed &other) {
  return !(a == other);
}
}}
#endif

