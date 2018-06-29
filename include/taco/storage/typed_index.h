#ifndef TACO_STORAGE_TYPED_INDEX_H
#define TACO_STORAGE_TYPED_INDEX_H

#include <taco/type.h>
#include <taco/storage/typed_value.h>

namespace taco {

class TypedIndexVal;
class TypedIndexRef;
class TypedIndexPtr;

/// Manipulate a dynamically typed value stored in an IndexTypeUnion.
/// This is separate from TypedComponent as indexes are allowed to be at most 64-bits even though TypedComponents store unions of size 128-bits. By using a separate class, we don't take a performance penalty.
/// TypedIndexVal and TypedIndexRef are wrappers around the implementations of these methods.
/// We do not use abstract methods to avoid the performance penalty.
/// NOTE: The implementations of these methods are very similar to TypedComponent in typed_value.h make sure to keep in sync.
class TypedIndex {
public:
  /// Gets the DataType of this TypedIndex
  const Datatype& getType() const;

protected:
  /// Gets the value of this TypedIndex as a size_t (for use in indexing)
  size_t getAsIndex(const IndexTypeUnion& mem) const;
  /// Sets mem to value (ensure that it does not write to bytes past the size of the type in the union)
  void set(IndexTypeUnion& mem, const IndexTypeUnion& value);
  /// Sets mem to casted value of integer
  void setInt(IndexTypeUnion& mem, const int value);
  /// Add the values of two IndexTypeUnions into a result
  void add(IndexTypeUnion& result, const IndexTypeUnion& a, const IndexTypeUnion& b) const;
  /// Add the values of one IndexTypeUnions with an integer constant into a result
  void addInt(IndexTypeUnion& result, const IndexTypeUnion& a, const int b) const;
  /// Multiply the values of two IndexTypeUnions into a result
  void multiply(IndexTypeUnion& result, const IndexTypeUnion& a, const IndexTypeUnion& b) const;
  /// Multiply the values of one IndexTypeUnions with an integer constant into a result
  void multiplyInt(IndexTypeUnion& result, const IndexTypeUnion& a, const int b) const;
  /// Multiply operator for two TypedIndexes
  TypedIndexVal operator*(const TypedIndex& other) const;

  /// DataType of TypedIndex
  Datatype dType;
};
  
/// Stores IndexTypeUnion and calls methods on TypedIndex with value
/// NOTE: The implementations of these methods are very similar to TypedComponentVal in typed_value.h make sure to keep in sync.
class TypedIndexVal: public TypedIndex {
public:
  /// Create an undefined type TypedIndexVal
  TypedIndexVal();
  /// Create a TypedIndexVal with DataType type
  TypedIndexVal(Datatype type);
  /// Create a TypedIndexVal initialized with the value and type of ref
  TypedIndexVal(TypedIndexRef ref);
  /// Create a TypedIndexVal initialized with the value stored at ptr of the size of DataType t
  TypedIndexVal(Datatype t, int constant) {
    dType = t;
    set(constant);
  }
  template<typename T>
  TypedIndexVal(Datatype t, T *ptr) {
    dType = t;
    TypedIndex::set(val, *((IndexTypeUnion *) ptr));
  }

  /// Gets a reference to the stored IndexTypeUnion
  IndexTypeUnion& get();
  /// Gets the value of the stored IndexTypeUnion
  IndexTypeUnion get() const;
  /// Gets the value of this TypedIndexVal as a size_t (for use in indexing)
  size_t getAsIndex() const;
  /// Sets the value to the value of a TypedIndexVal (must be same type)
  void set(TypedIndexVal value);
  /// Sets the value to the value of a TypedIndexRef (must be same type)
  void set(TypedIndexRef value);
  /// Sets the value to the value of a constant
  void set(int constant);
  /// Sets the value to the value of a TypedComponentVal (must be same type)
  void set(TypedComponentVal val);
  /// Sets the value to the value of a TypedComponentRef (must be same type)
  void set(TypedComponentRef val);

  /// Pre-increments the value
  TypedIndexVal operator++();
  /// Post-increments the value
  TypedIndexVal operator++(int junk);
  /// Adds two TypedIndexVals (must be same type)
  TypedIndexVal operator+(const TypedIndexVal other) const;
  /// Multiplies two TypedIndexVals (must be same type)
  TypedIndexVal operator*(const TypedIndexVal other) const;
  /// Adds a constant to a TypedIndexVal
  TypedIndexVal operator+(const int other) const;
  /// Multiplies TypedComponentVal by a constant
  TypedIndexVal operator*(const int other) const;
  /// Sets the value to the value of a constant
  TypedIndexVal operator=(const int other);

  /// Type of ptr for use when templating a class on TypedComponentVal and TypedIndexVal
  typedef TypedIndexPtr Ptr;
  /// Type of ref for use when templating a class on TypedComponentVal and TypedIndexVal
  typedef TypedIndexRef Ref;

private:
  /// Stored IndexTypeUnion that operations manipulate
  IndexTypeUnion val;
};


/// Pointer to a dynamically typed value. Dereferences to a TypedIndexRef.
/// Useful for doing pointer manipulations on dynamically sized values
/// NOTE: The implementations of these methods are very similar to TypedComponentPtr in typed_value.h make sure to keep in sync.
class TypedIndexPtr {
public:
  /// Creates a TypedIndexPtr with a nullptr
  TypedIndexPtr();
  /// Creates a TypedIndexPtr with a given type and memory location
  TypedIndexPtr (Datatype type, void *ptr);

  /// Gets the pointer stored by the TypedIndexPtr
  void* get();
  /// Dereferences the TypedIndexPtr to a TypedIndexRef
  TypedIndexRef operator*() const;

  /// Compare TypedIndexPtrs
  bool operator> (const TypedIndexPtr &other) const;
  /// Compare TypedIndexPtrs
  bool operator<= (const TypedIndexPtr &other) const;
  /// Compare TypedIndexPtrs
  bool operator< (const TypedIndexPtr &other) const;
  /// Compare TypedIndexPtrs
  bool operator>= (const TypedIndexPtr &other) const;
  /// Compare TypedIndexPtrs
  bool operator== (const TypedIndexPtr &other) const;
  /// Compare TypedIndexPtrs
  bool operator!= (const TypedIndexPtr &other) const;

  /// Increment the pointer by (the size of the datatype * value)
  TypedIndexPtr operator+(int value) const;
  /// Pre-increment the pointer by the size of the datatype
  TypedIndexPtr operator++();
  /// Post-increment the pointer by the size of the datatype
  TypedIndexPtr operator++(int junk);

  /// Type of val for use when templating a class on TypedComponentVal and TypedIndexVal
  typedef TypedIndexVal Val;
  /// Type of ref for use when templating a class on TypedComponentVal and TypedIndexVal
  typedef TypedIndexRef Ref;

private:
  /// Datatype of pointer
  Datatype type;
  /// Stored pointer to where dynamically typed value is
  void *ptr;
};

/// Reference to a IndexTypeUnion and calls methods on TypedIndex with dereferenced value.
/// Similar to TypedIndexVal.
/// NOTE: The implementations of these methods are very similar to TypedComponentRef in typed_value.h make sure to keep in sync.
class TypedIndexRef: public TypedIndex {
public:
  /// Create a TypedIndexRef initialized with the value stored at ptr of the size of DataType t
  template<typename T>
  TypedIndexRef(Datatype t, T *ptr) : ptr(reinterpret_cast<IndexTypeUnion *>(ptr)) {
    dType = t;
  }

  /// Dereferences to a TypedIndexPtr
  TypedIndexPtr operator&() const;
  /// Gets a reference to the stored IndexTypeUnion
  IndexTypeUnion& get();
  /// Gets the value of the stored IndexTypeUnion
  IndexTypeUnion get() const;
  /// Gets the value of this TypedIndexRef as a size_t (for use in indexing)
  size_t getAsIndex() const;
  /// Sets the value to the value of a TypedIndexVal (must be same type)
  void set(TypedIndexVal value);

  /// Sets the reference to a TypedIndexVal
  TypedIndexRef operator=(TypedIndexVal other);
  /// Sets the reference to the value of another TypedIndexRef
  TypedIndexRef operator=(TypedIndexRef other);
  /// Sets the reference to the value of a constant
  TypedIndexRef operator=(const int other);
  /// Pre-increments the reference
  TypedIndexRef operator++();
  /// Post-increments the reference
  TypedIndexRef operator++(int junk);
  /// Adds by TypedIndexVal with the result of a new TypedIndexVal (must be same type)
  TypedIndexVal operator+(const TypedIndexVal other) const;
  /// Multiplies by TypedIndexVal with the result of a new TypedIndexVal (must be same type)
  TypedIndexVal operator*(const TypedIndexVal other) const;
  /// Adds a constant to a TypedIndexRef with the result of a new TypedIndexVal
  TypedIndexVal operator+(const int other) const;
  /// Multiplies a constant to a TypedIndexRef with the result of a new TypedIndexVal
  TypedIndexVal operator*(const int other) const;

  /// Type of val for use when templating a class on TypedComponentVal and TypedIndexVal
  typedef TypedIndexVal Val;
  /// Type of ptr for use when templating a class on TypedComponentVal and TypedIndexVal
  typedef TypedIndexPtr Ptr;
private:
  /// Stored reference to IndexTypeUnion that operations manipulate
  IndexTypeUnion *ptr;
};

/// Compare two TypedIndexVal
bool operator>(const TypedIndexVal& a, const TypedIndexVal &other);
/// Compare two TypedIndexVal
bool operator==(const TypedIndexVal& a, const TypedIndexVal &other);
/// Compare two TypedIndexVal
bool operator>=(const TypedIndexVal& a,const TypedIndexVal &other);
/// Compare two TypedIndexVal
bool operator<(const TypedIndexVal& a, const TypedIndexVal &other);
/// Compare two TypedIndexVal
bool operator<=(const TypedIndexVal& a, const TypedIndexVal &other);
/// Compare two TypedIndexVal
bool operator!=(const TypedIndexVal& a, const TypedIndexVal &other);
/// Compare a TypedIndexVal with a constant
bool operator>(const TypedIndexVal& a, const int other);
/// Compare a TypedIndexVal with a constant
bool operator==(const TypedIndexVal& a, const int other);
/// Compare a TypedIndexVal with a constant
bool operator>=(const TypedIndexVal& a,const int other);
/// Compare a TypedIndexVal with a constant
bool operator<(const TypedIndexVal& a, const int other);
/// Compare a TypedIndexVal with a constant
bool operator<=(const TypedIndexVal& a, const int other);
/// Compare a TypedIndexVal with a constant
bool operator!=(const TypedIndexVal& a, const int other);



}
#endif

