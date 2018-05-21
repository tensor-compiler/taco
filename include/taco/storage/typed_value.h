#ifndef TACO_STORAGE_TYPED_VALUE_H
#define TACO_STORAGE_TYPED_VALUE_H

#include <taco/type.h>

namespace taco {
namespace storage {

class TypedComponentVal;
class TypedComponentRef;

// Holds a dynamically typed value
class TypedComponent {
public:
  const DataType& getType() const;
  size_t getAsIndex(const ValueTypeUnion mem) const;

  void set(ValueTypeUnion& mem, ValueTypeUnion value);
  void setInt(ValueTypeUnion& mem, const int value);

  void add(ValueTypeUnion& result, const ValueTypeUnion a, const ValueTypeUnion b) const;
  void addInt(ValueTypeUnion& result, const ValueTypeUnion a, const int b) const;
  void multiply(ValueTypeUnion& result, const ValueTypeUnion a, const ValueTypeUnion b) const;
  void multiplyInt(ValueTypeUnion& result, const ValueTypeUnion a, const int b) const;

  TypedComponentVal operator*(const TypedComponent& other) const;
protected:
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

