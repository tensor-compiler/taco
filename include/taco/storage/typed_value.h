#ifndef TACO_STORAGE_TYPED_VALUE_H
#define TACO_STORAGE_TYPED_VALUE_H

#include <taco/type.h>

namespace taco {
namespace storage {

class TypedComponentVal;
class TypedComponentRef;
class TypedComponentPtr;

/// Manipulate a dynamically typed value stored in a ValueTypeUnion.
/// TypedComponentVal and TypedComponentRef are wrappers around the implementations of these methods.
/// We do not use abstract methods to avoid the performance penalty.
/// NOTE: The implementations of these methods are very similar to TypedIndex in typed_index.h make sure to keep in sync.
class TypedComponent {
public:
  /// Gets the DataType of this TypedComponent
  const DataType& getType() const;
  /// Gets the value of this TypedComponent as a size_t (for use in indexing)
  size_t getAsIndex(const ValueTypeUnion mem) const;

protected:
  /// Sets mem to value (ensure that it does not write to bytes past the size of the type in the union)
  void set(ValueTypeUnion& mem, const ValueTypeUnion& value);
  /// Sets mem to casted value of integer
  void setInt(ValueTypeUnion& mem, const int value);
  /// Add the values of two ValueTypeUnion into a result
  void add(ValueTypeUnion& result, const ValueTypeUnion& a, const ValueTypeUnion& b) const;
  /// Add the values of one ValueTypeUnion with an integer constant into a result
  void addInt(ValueTypeUnion& result, const ValueTypeUnion& a, const int b) const;
  /// Multiply the values of two ValueTypeUnion into a result
  void multiply(ValueTypeUnion& result, const ValueTypeUnion& a, const ValueTypeUnion& b) const;
  /// Multiply the values of one ValueTypeUnion with an integer constant into a result
  void multiplyInt(ValueTypeUnion& result, const ValueTypeUnion& a, const int b) const;
  /// Multiply operator for two TypedComponents
  TypedComponentVal operator*(const TypedComponent& other) const;

  /// DataType of TypedComponent
  DataType dType;
};

// Allocates a union to hold a dynamically typed value
class TypedComponentVal: public TypedComponent {
public:
  TypedComponentVal();
  TypedComponentVal(DataType type);
  TypedComponentVal(TypedComponentRef ref);

  TypedComponentVal(DataType t, int constant) {
    dType = t;
    set(constant);
  }

  template<typename T>
  TypedComponentVal(DataType t, T *ptr) {
    dType = t;
    TypedComponent::set(val, *((ValueTypeUnion *) ptr));
  }

  ValueTypeUnion& get();

  ValueTypeUnion get() const;

  const DataType& getType() const;

  size_t getAsIndex() const;

  void set(TypedComponentVal value);

  void set(TypedComponentRef value);

  void set(int constant);

  TypedComponentVal operator++();

  TypedComponentVal operator++(int junk);

  TypedComponentVal operator+(const TypedComponentVal other) const;

  TypedComponentVal operator*(const TypedComponentVal other) const;

  TypedComponentVal operator+(const int other) const;

  TypedComponentVal operator*(const int other) const;

  TypedComponentVal operator=(const int other);

  typedef TypedComponentPtr Ptr;
  typedef TypedComponentRef Ref;

private:
  ValueTypeUnion val;
};


// dereferences to typedref
class TypedComponentPtr {
public:
  TypedComponentPtr() : ptr(nullptr) {}

  TypedComponentPtr (DataType type, void *ptr) : type(type), ptr(ptr) {
  }

  void* get();

  TypedComponentRef operator*() const;
  
  bool operator> (const TypedComponentPtr &other) const;
  bool operator<= (const TypedComponentPtr &other) const;

  bool operator< (const TypedComponentPtr &other) const;
  bool operator>= (const TypedComponentPtr &other) const;

  bool operator== (const TypedComponentPtr &other) const;
  bool operator!= (const TypedComponentPtr &other) const;

  TypedComponentPtr operator+(int value) const;
  TypedComponentPtr operator++();
  TypedComponentPtr operator++(int junk);

  typedef TypedComponentVal Val;
  typedef TypedComponentRef Ref;

private:
  DataType type;
  void *ptr;
};

class TypedComponentRef: public TypedComponent{
public:
  template<typename T>
  TypedComponentRef(DataType t, T *ptr) : ptr(reinterpret_cast<ValueTypeUnion *>(ptr)) {
    dType = t;
  }

  ValueTypeUnion& get();

  ValueTypeUnion get() const;

  TypedComponentPtr operator&() const;

  void set(TypedComponentVal value);

  TypedComponentRef operator=(TypedComponentVal other);

  TypedComponentRef operator=(TypedComponentRef other);

  TypedComponentRef operator++();

  TypedComponentRef operator++(int junk);

  TypedComponentVal operator+(const TypedComponentVal other) const;

  TypedComponentVal operator*(const TypedComponentVal other) const;

  TypedComponentVal operator+(const int other) const;

  TypedComponentVal operator*(const int other) const;

  TypedComponentRef operator=(const int other);

  const DataType& getType() const;

  size_t getAsIndex() const;

  typedef TypedComponentVal Val;
  typedef TypedComponentPtr Ptr;

private:
  ValueTypeUnion *ptr;
};


bool operator>(const TypedComponentVal& a, const TypedComponentVal &other);

bool operator==(const TypedComponentVal& a, const TypedComponentVal &other);

bool operator>=(const TypedComponentVal& a,const TypedComponentVal &other);

bool operator<(const TypedComponentVal& a, const TypedComponentVal &other);

bool operator<=(const TypedComponentVal& a, const TypedComponentVal &other);

bool operator!=(const TypedComponentVal& a, const TypedComponentVal &other);

bool operator>(const TypedComponentVal& a, const int other);

bool operator==(const TypedComponentVal& a, const int other);

bool operator>=(const TypedComponentVal& a,const int other);

bool operator<(const TypedComponentVal& a, const int other);

bool operator<=(const TypedComponentVal& a, const int other);

bool operator!=(const TypedComponentVal& a, const int other);

}}
#endif

