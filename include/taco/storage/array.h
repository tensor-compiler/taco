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
  /// Allocates a memory location
  TypedValue();
  TypedValue(DataType type);

  template<typename T>
  TypedValue(DataType type, T constant) : type(type), memLocation(malloc(type.getNumBytes())), memAllocced(true) {
    set(constant);
  }

  template<class T>
  TypedValue(DataType type, T *memLocation) : type(type), memLocation(memLocation), memAllocced(false) {
  }

  TypedValue(const TypedValue& other); // copy constructor
  TypedValue(TypedValue&& other); // move constructor
  ~TypedValue(); //destructor

  const DataType& getType() const;

  /// Returns a pointer to the memory location
  void* get() const;

  unsigned long long getAsIndex() const;

  void set(TypedValue value);

  //Casts constant to type
  template<typename T>
  void set(T constant) {
    switch (type.getKind()) {
      case DataType::Bool: *((bool *) memLocation) = (bool) constant; break;
      case DataType::UInt8: *((uint8_t *) memLocation) = (uint8_t) constant; break;
      case DataType::UInt16: *((uint16_t *) memLocation) = (uint16_t) constant; break;
      case DataType::UInt32: *((uint32_t *) memLocation) = (uint32_t) constant; break;
      case DataType::UInt64: *((uint64_t *) memLocation) = (uint64_t) constant; break;
      case DataType::UInt128: *((unsigned long long *) memLocation) = (unsigned long long) constant; break;
      case DataType::Int8: *((int8_t *) memLocation) = (int8_t) constant; break;
      case DataType::Int16: *((int16_t *) memLocation) = (int16_t) constant; break;
      case DataType::Int32: *((int32_t *) memLocation) = (int32_t) constant; break;
      case DataType::Int64: *((int64_t *) memLocation) = (int64_t) constant; break;
      case DataType::Int128: *((long long *) memLocation) = (long long) constant; break;
      case DataType::Float32: *((float *) memLocation) = (float) constant; break;
      case DataType::Float64: *((double *) memLocation) = (double) constant; break;
      case DataType::Complex64: taco_ierror; break;
      case DataType::Complex128: taco_ierror; break;
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
    TypedValue test = TypedValue(type);
    test.set(other);
    return *this > test;
  }

  template<class T>
  bool operator<= (T other) const {
    TypedValue test = TypedValue(type);
    test.set(other);
    return *this <= test;
  }

  template<class T>
  bool operator< (T other) const {
    TypedValue test = TypedValue(type);
    test.set(other);
    return *this < test;
  }

  template<class T>
  bool operator>= (T other) const {
    TypedValue test = TypedValue(type);
    test.set(other);
    return *this >= test;
  }

  template<class T>
  bool operator!= (T other) const {
    TypedValue test = TypedValue(type);
    test.set(other);
    return *this != test;
  }

  template<class T>
  bool operator== (T other) const {
    TypedValue test = TypedValue(type);
    test.set(other);
    return *this == test;
  }

  TypedValue operator+(const TypedValue &other) const;

  template<class T>
  TypedValue operator+(T other) {
    return *this + TypedValue(type, other);
  }

  TypedValue operator*(const TypedValue &other) const;

  template<class T>
  TypedValue operator*(int other) const {
    return *this * TypedValue(type, other);
  }

  TypedValue& operator=(const TypedValue& other); //copy assignment operator
  TypedValue& operator=(TypedValue&& other); //move assignment operator

  template<class T>
  TypedValue operator=(T other) {
    set(other);
    return *this;
  }

  TypedValue operator++();

private:
  DataType type;
  void *memLocation;
  void set(void *location);
  bool memAllocced;
  void cleanupMemory();
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

