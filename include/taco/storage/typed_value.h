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

protected:
  /// Gets the value of this TypedComponent as a size_t (for use in indexing)
  size_t getAsIndex(const ValueTypeUnion mem) const;
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

  /// DataType of TypedComponent
  DataType dType;
};

/// Stores ValueTypeUnion and calls methods on TypedComponent with value
/// NOTE: The implementations of these methods are very similar to TypedIndexVal in typed_index.h make sure to keep in sync.
class TypedComponentVal: public TypedComponent {
public:
  /// Create an undefined type TypedComponentVal
  TypedComponentVal();
  /// Create a TypedComponentVal with DataType type
  TypedComponentVal(DataType type);
  /// Create a TypedComponentVal initialized with the value and type of ref
  TypedComponentVal(TypedComponentRef ref);
  /// Create a TypedComponentVal initialized with type t and the value of constant
  TypedComponentVal(DataType t, int constant);
  /// Create a TypedComponentVal initialized with the value stored at ptr of the size of DataType t
  template<typename T>
  TypedComponentVal(DataType t, T *ptr) {
    dType = t;
    TypedComponent::set(val, *((ValueTypeUnion *) ptr));
  }

  /// Gets a reference to the stored ValueTypeUnion
  ValueTypeUnion& get();
  /// Gets the value of the stored ValueTypeUnion
  ValueTypeUnion get() const;
  /// Gets the value of this TypedComponent as a size_t (for use in indexing)
  size_t getAsIndex() const;
  /// Sets the value to the value of a TypedComponentVal (must be same type)
  void set(TypedComponentVal value);
  /// Sets the value to the value of a TypedComponentRef (must be same type)
  void set(TypedComponentRef value);
  /// Sets the value to the value of a constant
  void set(int constant);

  TypedComponentVal operator++();
  /// Post-increments the value
  TypedComponentVal operator++(int junk);
  /// Adds two TypedComponentVals (must be same type)
  TypedComponentVal operator+(const TypedComponentVal other) const;
  /// Multiplies two TypedComponentVals (must be same type)
  TypedComponentVal operator*(const TypedComponentVal other) const;
  /// Adds a constant to a TypedComponentVal
  TypedComponentVal operator+(const int other) const;
  /// Multiplies TypedComponentVal by a constant
  TypedComponentVal operator*(const int other) const;
  /// Sets the value to the value of a constant
  TypedComponentVal operator=(const int other);

  /// Type of ptr for use when templating a class on TypedComponentVal and TypedIndexVal
  typedef TypedComponentPtr Ptr;
  /// Type of ref for use when templating a class on TypedComponentVal and TypedIndexVal
  typedef TypedComponentRef Ref;

private:
  /// Stored ValueTypeUnion that operations manipulate
  ValueTypeUnion val;
};


/// Pointer to a dynamically typed value. Dereferences to a TypedComponentRef.
/// Useful for doing pointer manipulations on dynamically sized values
/// NOTE: The implementations of these methods are very similar to TypedIndexPtr in typed_index.h make sure to keep in sync.
class TypedComponentPtr {
public:
  /// Creates a TypedComponentPtr with a nullptr
  TypedComponentPtr();
  /// Creates a TypedComponentPtr with a given type and memory location
  TypedComponentPtr (DataType type, void *ptr);

  /// Gets the pointer stored by the TypedComponentPtr
  void* get();
  /// Dereferences the TypedComponent to a TypedComponentRef
  TypedComponentRef operator*() const;

  /// Compare TypedComponentPtrs
  bool operator> (const TypedComponentPtr &other) const;
  /// Compare TypedComponentPtrs
  bool operator<= (const TypedComponentPtr &other) const;
  /// Compare TypedComponentPtrs
  bool operator< (const TypedComponentPtr &other) const;
  /// Compare TypedComponentPtrs
  bool operator>= (const TypedComponentPtr &other) const;
  /// Compare TypedComponentPtrs
  bool operator== (const TypedComponentPtr &other) const;
  /// Compare TypedComponentPtrs
  bool operator!= (const TypedComponentPtr &other) const;

  /// Increment the pointer by (the size of the datatype * value)
  TypedComponentPtr operator+(int value) const;
  /// Pre-increment the pointer by the size of the datatype
  TypedComponentPtr operator++();
  /// Post-increment the pointer by the size of the datatype
  TypedComponentPtr operator++(int junk);

  /// Type of val for use when templating a class on TypedComponentVal and TypedIndexVal
  typedef TypedComponentVal Val;
  /// Type of ref for use when templating a class on TypedComponentVal and TypedIndexVal
  typedef TypedComponentRef Ref;

private:
  DataType type;
  void *ptr;
};

/// Reference to a ValueTypeUnion and calls methods on TypedComponent with dereferenced value.
/// Similar to TypedComponentVal.
/// NOTE: The implementations of these methods are very similar to TypedIndexVal in typed_index.h make sure to keep in sync.
class TypedComponentRef: public TypedComponent {
public:
  /// Create a TypedComponentRef initialized with the value stored at ptr of the size of DataType t
  template<typename T>
  TypedComponentRef(DataType t, T *ptr) : ptr(reinterpret_cast<ValueTypeUnion *>(ptr)) {
    dType = t;
  }

  /// Dereferences to a TypedComponentPtr
  TypedComponentPtr operator&() const;
  /// Gets a reference to the stored ValueTypeUnion
  ValueTypeUnion& get();
  /// Gets the value of the stored ValueTypeUnion
  ValueTypeUnion get() const;
  /// Gets the value of this TypedComponent as a size_t (for use in indexing)
  size_t getAsIndex() const;
  /// Sets the value to the value of a TypedComponentVal (must be same type)
  void set(TypedComponentVal value);

  /// Sets the reference to a TypedComponentVal
  TypedComponentRef operator=(TypedComponentVal other);
  /// Sets the reference to the value of another TypedComponentRef
  TypedComponentRef operator=(TypedComponentRef other);
  /// Sets the reference to the value of a constant
  TypedComponentRef operator=(const int other);
  /// Pre-increments the reference
  TypedComponentRef operator++();
  /// Post-increments the reference
  TypedComponentRef operator++(int junk);
  /// Adds two TypedComponentRefs with the result of a new TypedComponentVal (must be same type)
  TypedComponentVal operator+(const TypedComponentVal other) const;
  /// Multiplies two TypedComponentRefs with the result of a new TypedComponentVal (must be same type)
  TypedComponentVal operator*(const TypedComponentVal other) const;
  /// Adds a constant to a TypedComponentRef with the result of a new TypedComponentVal
  TypedComponentVal operator+(const int other) const;
  /// Multiplies a constant to a TypedComponentRef with the result of a new TypedComponentVal
  TypedComponentVal operator*(const int other) const;

  /// Type of val for use when templating a class on TypedComponentVal and TypedIndexVal
  typedef TypedComponentVal Val;
  /// Type of ptr for use when templating a class on TypedComponentVal and TypedIndexVal
  typedef TypedComponentPtr Ptr;

private:
  /// Stored reference to ValueTypeUnion that operations manipulate
  ValueTypeUnion *ptr;
};

/// Compare two TypedComponentVals
bool operator>(const TypedComponentVal& a, const TypedComponentVal &other);
/// Compare two TypedComponentVals
bool operator==(const TypedComponentVal& a, const TypedComponentVal &other);
/// Compare two TypedComponentVals
bool operator>=(const TypedComponentVal& a,const TypedComponentVal &other);
/// Compare two TypedComponentVals
bool operator<(const TypedComponentVal& a, const TypedComponentVal &other);
/// Compare two TypedComponentVals
bool operator<=(const TypedComponentVal& a, const TypedComponentVal &other);
/// Compare two TypedComponentVals
bool operator!=(const TypedComponentVal& a, const TypedComponentVal &other);
/// Compare a TypedComponentVal with a constant
bool operator>(const TypedComponentVal& a, const int other);
/// Compare a TypedComponentVal with a constant
bool operator==(const TypedComponentVal& a, const int other);
/// Compare a TypedComponentVal with a constant
bool operator>=(const TypedComponentVal& a,const int other);
/// Compare a TypedComponentVal with a constant
bool operator<(const TypedComponentVal& a, const int other);
/// Compare a TypedComponentVal with a constant
bool operator<=(const TypedComponentVal& a, const int other);
/// Compare a TypedComponentVal with a constant
bool operator!=(const TypedComponentVal& a, const int other);

}}
#endif

