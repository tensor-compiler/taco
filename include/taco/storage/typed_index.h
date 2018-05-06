#ifndef TACO_STORAGE_TYPED_INDEX_H
#define TACO_STORAGE_TYPED_INDEX_H

#include <taco/type.h>
#include <taco/storage/typed_value.h>

namespace taco {
namespace storage {

class TypedIndex;
class TypedIndexRef;

// Holds a dynamically typed index
class TypedI {
public:
  const DataType& getType() const;
  size_t getAsIndex(const IndexTypeUnion mem) const;

  void set(IndexTypeUnion& mem, IndexTypeUnion value);
  void setInt(IndexTypeUnion& mem, const int value);

  void add(IndexTypeUnion& result, const IndexTypeUnion a, const IndexTypeUnion b) const;
  void addInt(IndexTypeUnion& result, const IndexTypeUnion a, const int b) const;
  void multiply(IndexTypeUnion& result, const IndexTypeUnion a, const IndexTypeUnion b) const;
  void multiplyInt(IndexTypeUnion& result, const IndexTypeUnion a, const int b) const;

  TypedIndex operator*(const TypedI& other) const;
protected:
  DataType dType;
};

  
// Allocates a union to hold a dynamically typed value
class TypedIndex: public TypedI {
public:
  TypedIndex();
  TypedIndex(DataType type);
  TypedIndex(TypedIndexRef ref);


  TypedIndex(DataType t, int constant) {
    dType = t;
    set(constant);
  }

  template<typename T>
  TypedIndex(DataType t, T *ptr) {
    dType = t;
    TypedI::set(val, *((IndexTypeUnion *) ptr));
  }

  IndexTypeUnion& get();

  IndexTypeUnion get() const;

  const DataType& getType() const;

  size_t getAsIndex() const;

  void set(TypedIndex value);

  void set(TypedIndexRef value);

  void set(int constant);

  void set(TypedValue val);
  void set(TypedRef val);

  TypedIndex operator++();

  TypedIndex operator++(int junk);

  TypedIndex operator+(const TypedIndex other) const;

  TypedIndex operator*(const TypedIndex other) const;

  TypedIndex operator+(const int other) const;

  TypedIndex operator*(const int other) const;

  TypedIndex operator=(const int other);

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

class TypedIndexRef: public TypedI {
public:
  template<typename T>
  TypedIndexRef(DataType t, T *ptr) : ptr(reinterpret_cast<IndexTypeUnion *>(ptr)) {
    dType = t;
  }

  IndexTypeUnion& get();

  IndexTypeUnion get() const;

  TypedIndexPtr operator&() const;

  void set(TypedIndex value);

  TypedIndexRef operator=(TypedIndex other);

  TypedIndexRef operator=(TypedIndexRef other);

  TypedIndexRef operator++();

  TypedIndexRef operator++(int junk);

  TypedIndex operator+(const TypedIndex other) const;

  TypedIndex operator*(const TypedIndex other) const;

  TypedIndex operator+(const int other) const;

  TypedIndex operator*(const int other) const;

  TypedIndexRef operator=(const int other);

  const DataType& getType() const;

  size_t getAsIndex() const;


private:
  IndexTypeUnion *ptr;
};


bool operator>(const TypedIndex& a, const TypedIndex &other);

bool operator==(const TypedIndex& a, const TypedIndex &other);

bool operator>=(const TypedIndex& a,const TypedIndex &other);

bool operator<(const TypedIndex& a, const TypedIndex &other);

bool operator<=(const TypedIndex& a, const TypedIndex &other);

bool operator!=(const TypedIndex& a, const TypedIndex &other);

bool operator>(const TypedIndex& a, const int other);

bool operator==(const TypedIndex& a, const int other);

bool operator>=(const TypedIndex& a,const int other);

bool operator<(const TypedIndex& a, const int other);

bool operator<=(const TypedIndex& a, const int other);

bool operator!=(const TypedIndex& a, const int other);



}}
#endif

