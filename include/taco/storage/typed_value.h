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

  void set(DataTypeUnion& mem, DataTypeUnion value);
  void setInt(DataTypeUnion& mem, const int value);

  void add(DataTypeUnion& result, const DataTypeUnion a, const DataTypeUnion b) const;
  void addInt(DataTypeUnion& result, const DataTypeUnion a, const int b) const;
  void multiply(DataTypeUnion& result, const DataTypeUnion a, const DataTypeUnion b) const;
  void multiplyInt(DataTypeUnion& result, const DataTypeUnion a, const int b) const;

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

  TypedValue(DataType t, int constant) {
    dType = t;
    set(constant);
  }

  template<typename T>
  TypedValue(DataType t, T *ptr) {
    dType = t;
    Typed::set(val, *((DataTypeUnion *) ptr));
  }

  DataTypeUnion& get();

  DataTypeUnion get() const;

  const DataType& getType() const;

  size_t getAsIndex() const;

  void set(TypedValue value);

  void set(TypedRef value);

  void set(int constant);

  TypedValue operator++();

  TypedValue operator++(int junk);

  TypedValue operator+(const TypedValue other) const;

  TypedValue operator*(const TypedValue other) const;

  TypedValue operator+(const int other) const;

  TypedValue operator*(const int other) const;

  TypedValue operator=(const int other);



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

  DataTypeUnion& get();

  DataTypeUnion get() const;

  TypedPtr operator&() const;

  void set(TypedValue value);

  TypedRef operator=(TypedValue other);

  TypedRef operator=(TypedRef other);

  TypedRef operator++();

  TypedRef operator++(int junk);

  TypedValue operator+(const TypedValue other) const;

  TypedValue operator*(const TypedValue other) const;

  TypedValue operator+(const int other) const;

  TypedValue operator*(const int other) const;

  TypedRef operator=(const int other);

  const DataType& getType() const;

  size_t getAsIndex() const;


private:
  DataTypeUnion *ptr;
};


bool operator>(const TypedValue& a, const TypedValue &other);

bool operator==(const TypedValue& a, const TypedValue &other);

bool operator>=(const TypedValue& a,const TypedValue &other);

bool operator<(const TypedValue& a, const TypedValue &other);

bool operator<=(const TypedValue& a, const TypedValue &other);

bool operator!=(const TypedValue& a, const TypedValue &other);

bool operator>(const TypedValue& a, const int other);

bool operator==(const TypedValue& a, const int other);

bool operator>=(const TypedValue& a,const int other);

bool operator<(const TypedValue& a, const int other);

bool operator<=(const TypedValue& a, const int other);

bool operator!=(const TypedValue& a, const int other);

}}
#endif

