#ifndef TACO_STORAGE_TYPED_INDEX_H
#define TACO_STORAGE_TYPED_INDEX_H

#include <taco/type.h>
#include <taco/storage/typed_value.h>

namespace taco {
namespace storage {

class TypedIndexVal;
class TypedIndexRef;
class TypedIndexPtr;

/// Manipulate a dynamically typed value stored in an IndexTypeUnion.
/// TypedIndexVal and TypedIndexRef are wrappers around the implementations of these methods.
/// We do not use abstract methods to avoid the performance penalty.
/// NOTE: The implementations of these methods are very similar to TypedComponent in typed_value.h make sure to keep in sync.
class TypedIndex {
public:
  /// Gets the DataType of this TypedIndex
  const DataType& getType() const;
  /// Gets the value of this TypedIndex as a size_t (for use in indexing)
  size_t getAsIndex(const IndexTypeUnion& mem) const;

protected:
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
  DataType dType;
};

  
// Allocates a union to hold a dynamically typed value
class TypedIndexVal: public TypedIndex {
public:
  TypedIndexVal();
  TypedIndexVal(DataType type);
  TypedIndexVal(TypedIndexRef ref);


  TypedIndexVal(DataType t, int constant) {
    dType = t;
    set(constant);
  }

  template<typename T>
  TypedIndexVal(DataType t, T *ptr) {
    dType = t;
    TypedIndex::set(val, *((IndexTypeUnion *) ptr));
  }

  IndexTypeUnion& get();

  IndexTypeUnion get() const;

  const DataType& getType() const;

  size_t getAsIndex() const;

  void set(TypedIndexVal value);

  void set(TypedIndexRef value);

  void set(int constant);

  void set(TypedComponentVal val);
  void set(TypedComponentRef val);

  TypedIndexVal operator++();

  TypedIndexVal operator++(int junk);

  TypedIndexVal operator+(const TypedIndexVal other) const;

  TypedIndexVal operator*(const TypedIndexVal other) const;

  TypedIndexVal operator+(const int other) const;

  TypedIndexVal operator*(const int other) const;

  TypedIndexVal operator=(const int other);

  typedef TypedIndexPtr Ptr;
  typedef TypedIndexRef Ref;
private:
  IndexTypeUnion val;
};


  // dereferences to TypedIndexRef
class TypedIndexPtr {
public:
  TypedIndexPtr() : ptr(nullptr) {}

  TypedIndexPtr (DataType type, void *ptr) : type(type), ptr(ptr) {
  }

  void* get();

  TypedIndexRef operator*() const;

  bool operator> (const TypedIndexPtr &other) const;
  bool operator<= (const TypedIndexPtr &other) const;

  bool operator< (const TypedIndexPtr &other) const;
  bool operator>= (const TypedIndexPtr &other) const;

  bool operator== (const TypedIndexPtr &other) const;
  bool operator!= (const TypedIndexPtr &other) const;

  TypedIndexPtr operator+(int value) const;
  TypedIndexPtr operator++();
  TypedIndexPtr operator++(int junk);

  typedef TypedIndexVal Val;
  typedef TypedIndexRef Ref;
private:
  DataType type;
  void *ptr;
};

class TypedIndexRef: public TypedIndex {
public:
  template<typename T>
  TypedIndexRef(DataType t, T *ptr) : ptr(reinterpret_cast<IndexTypeUnion *>(ptr)) {
    dType = t;
  }

  IndexTypeUnion& get();

  IndexTypeUnion get() const;

  TypedIndexPtr operator&() const;

  void set(TypedIndexVal value);

  TypedIndexRef operator=(TypedIndexVal other);

  TypedIndexRef operator=(TypedIndexRef other);

  TypedIndexRef operator++();

  TypedIndexRef operator++(int junk);

  TypedIndexVal operator+(const TypedIndexVal other) const;

  TypedIndexVal operator*(const TypedIndexVal other) const;

  TypedIndexVal operator+(const int other) const;

  TypedIndexVal operator*(const int other) const;

  TypedIndexRef operator=(const int other);

  const DataType& getType() const;

  size_t getAsIndex() const;

  typedef TypedIndexPtr Ptr;
  typedef TypedIndexVal Val;
private:
  IndexTypeUnion *ptr;
};


bool operator>(const TypedIndexVal& a, const TypedIndexVal &other);

bool operator==(const TypedIndexVal& a, const TypedIndexVal &other);

bool operator>=(const TypedIndexVal& a,const TypedIndexVal &other);

bool operator<(const TypedIndexVal& a, const TypedIndexVal &other);

bool operator<=(const TypedIndexVal& a, const TypedIndexVal &other);

bool operator!=(const TypedIndexVal& a, const TypedIndexVal &other);

bool operator>(const TypedIndexVal& a, const int other);

bool operator==(const TypedIndexVal& a, const int other);

bool operator>=(const TypedIndexVal& a,const int other);

bool operator<(const TypedIndexVal& a, const int other);

bool operator<=(const TypedIndexVal& a, const int other);

bool operator!=(const TypedIndexVal& a, const int other);



}}
#endif

