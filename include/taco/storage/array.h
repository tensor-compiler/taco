#ifndef TACO_STORAGE_ARRAY_H
#define TACO_STORAGE_ARRAY_H

#include <memory>
#include <ostream>
#include <taco/type.h>

namespace taco {
namespace storage {

/// Allows for performing certain operations on dynamically typed value
class TypedValue {
public:
  TypedValue();
  TypedValue(DataType type);

  template<typename T>
  TypedValue(DataType type, T constant) : type(type) {
    set(constant);
  }

  template<typename T>
  TypedValue(DataType type, T *ptr) : type(type) {
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
      case DataType::Complex64: set(*((std::complex<float>*) ptr)); break;
      case DataType::Complex128: set(*((std::complex<double>*) ptr)); break;
      case DataType::Undefined: taco_ierror; break;
    }
  }

  const DataType& getType() const;

  /// Returns a pointer to the memory location
  DataTypeUnion get() const;

  size_t getAsIndex() const;

  void set(TypedValue value);

  void set(DataTypeUnion value);

  //Casts constant to type
  template<typename T>
  void set(T constant) {
    switch (type.getKind()) {
      case DataType::Bool: val.boolValue = (bool) constant; break;
      case DataType::UInt8: val.uint8Value = (uint8_t) constant; break;
      case DataType::UInt16: val.uint16Value = (uint16_t) constant; break;
      case DataType::UInt32: val.uint32Value = (uint32_t) constant; break;
      case DataType::UInt64: val.uint64Value = (uint64_t) constant; break;
      case DataType::UInt128: val.uint128Value = (unsigned long long) constant; break;
      case DataType::Int8: val.int8Value = (int8_t) constant; break;
      case DataType::Int16: val.int16Value = (int16_t) constant; break;
      case DataType::Int32: val.int32Value = (int32_t) constant; break;
      case DataType::Int64: val.int64Value = (int64_t) constant; break;
      case DataType::Int128: val.int128Value = (long long) constant; break;
      case DataType::Float32: val.float32Value = (float) constant; break;
      case DataType::Float64: val.float64Value = (double) constant; break;
      case DataType::Complex64: taco_ierror; break; //explicit specialization
      case DataType::Complex128: taco_ierror; break; //explicit specialization
      case DataType::Undefined: taco_ierror; break;
    }
  }



  bool operator> (const TypedValue &other) const;
  bool operator<= (const TypedValue &other) const;

  bool operator< (const TypedValue &other) const;
  bool operator>= (const TypedValue &other) const;

  bool operator== (const TypedValue &other) const;
  bool operator!= (const TypedValue &other) const;

  template<class T>
  bool operator> (T other) const {
    TypedValue test(type);
    test.set(other);
    return *this > test;
  }

  template<class T>
  bool operator<= (T other) const {
    TypedValue test(type);
    test.set(other);
    return *this <= test;
  }

  template<class T>
  bool operator< (T other) const {
    TypedValue test(type);
    test.set(other);
    return *this < test;
  }

  template<class T>
  bool operator>= (T other) const {
    TypedValue test(type);
    test.set(other);
    return *this >= test;
  }

  template<class T>
  bool operator!= (T other) const {
    TypedValue test(type);
    test.set(other);
    return *this != test;
  }

  template<class T>
  bool operator== (T other) const {
    TypedValue test(type);
    test.set(other);
    return *this == test;
  }

  template<class T>
  TypedValue operator+(const T other) const {
    return *this + TypedValue(type, other);
  }

  TypedValue operator+(const TypedValue other) const;

  TypedValue operator++();

  template<class T>
  TypedValue operator*(const T other) const {
    return *this * TypedValue(type, other);
  }

  TypedValue operator*(const TypedValue other) const;

  template<class T>
  TypedValue operator=(T other) {
    set(other);
    return *this;
  }
private:
  DataType type;
  DataTypeUnion val;
};

template<>
void TypedValue::set(std::complex<float> constant) {
  val.complex64Value = (std::complex<float>) constant;
}

template<>
void TypedValue::set(std::complex<double> constant) {
  val.complex128Value = (std::complex<double>) constant;
}

class TypedRef {
public:
  TypedRef(DataType type, void *ptr) : type(type), ptr(ptr) {
  }

  TypedValue operator*() const {
    return TypedValue(type, ptr);
  }
  bool operator> (const TypedRef &other) const;
  bool operator<= (const TypedRef &other) const;

  bool operator< (const TypedRef &other) const;
  bool operator>= (const TypedRef &other) const;

  bool operator== (const TypedRef &other) const;
  bool operator!= (const TypedRef &other) const;

  TypedRef operator+(int value) const;
private:
  DataType type;
  void *ptr;
};



/// An array is a smart pointer to raw memory together with an element type,
/// a size (number of elements) and a reclamation policy.
class Array {
public:
  /// The memory reclamation policy of Array objects. UserOwns means the Array
  /// object will not free its data, free means it will reclaim data  with the
  /// C free function and delete means it will reclaim data with delete[].
  enum Policy {UserOwns, Free, Delete};

  /// Construct an empty array of undefined elements.
  Array();

  /// Construct an array of elements of the given type.
  Array(DataType type, void* data, size_t size, Policy policy=Free);

  /// Returns the type of the array elements
  const DataType& getType() const;

  /// Returns the number of array elements
  size_t getSize() const;

  /// Returns the array data.
  /// @{
  const void* getData() const;
  void* getData();
  /// @}

  TypedValue get(int index) const;
  TypedValue operator[] (const int index) const;

  /// Zero the array content
  void zero();

private:
  struct Content;
  std::shared_ptr<Content> content;
};

/// Send the array data as text to a stream.
std::ostream& operator<<(std::ostream&, const Array&);
std::ostream& operator<<(std::ostream&, Array::Policy);
}}
#endif

