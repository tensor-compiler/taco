#ifndef TACO_STORAGE_TYPED_INDEX_H
#define TACO_STORAGE_TYPED_INDEX_H

#include <taco/type.h>
#include <taco/storage/typed_value.h>

namespace taco {
namespace storage {

class TypedIndexVal;
class TypedIndexRef;

// Holds a dynamically typed index
class TypedIndex {
public:
  const DataType& getType() const;
  size_t getAsIndex(const IndexTypeUnion mem) const;

  void set(IndexTypeUnion& mem, IndexTypeUnion value);
  void setInt(IndexTypeUnion& mem, const int value);

  void add(IndexTypeUnion& result, const IndexTypeUnion a, const IndexTypeUnion b) const;
  void addInt(IndexTypeUnion& result, const IndexTypeUnion a, const int b) const;
  void multiply(IndexTypeUnion& result, const IndexTypeUnion a, const IndexTypeUnion b) const;
  void multiplyInt(IndexTypeUnion& result, const IndexTypeUnion a, const int b) const;

  TypedIndexVal operator*(const TypedIndex& other) const;
protected:
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

