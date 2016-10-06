#ifndef TACO_UTIL_INTRUSIVE_PTR_H
#define TACO_UTIL_INTRUSIVE_PTR_H

#include <iostream>

namespace taco {
namespace util {

/// Forward declare acquire and release methods
/// @{
template<typename T> void acquire(const T *);
template<typename T> void release(const T *);
/// @}

/// This class provides an intrusive pointer, which is a pointer that stores its
/// reference count in the managed class.  The managed class must therefore have
/// a reference count field and provide two functions 'acquire' and 'release'
/// to acquire and release a reference on itself.
///
/// For example:
/// struct X {
///   mutable long ref = 0;
///   friend void acquire(const X *x) { ++x->ref; }
///   friend void release(const X *x) { if (--x->ref ==0) delete x; }
/// };
template <class T>
class IntrusivePtr {
public:
  T *ptr;

  /// Allocate an undefined IntrusivePtr
  IntrusivePtr() : ptr(nullptr) {}

  /// Allocate an IntrusivePtr with an object
  IntrusivePtr(T *p) : ptr(p) {
    if (ptr) {
      acquire(ptr);
    }
  }

  /// Copy constructor
  IntrusivePtr(const IntrusivePtr &other) : ptr(other.ptr) {
    if (ptr) {
      acquire(ptr);
    }
  }

  /// Move constructor
  IntrusivePtr(IntrusivePtr &&other) : ptr(other.ptr) {
    other.ptr = nullptr;
  }

  /// Copy assignment operator
  IntrusivePtr& operator=(const IntrusivePtr &other) {
    if (ptr) {
      release(ptr);
    }
    ptr = other.ptr;
    if (ptr) {
      acquire(ptr);
    }
    return *this;
  }

  /// Copy assignment operator for managed object
  IntrusivePtr& operator=(T *p) {
    if (ptr) {
      release(ptr);
    }
    this->ptr = p;
    if (ptr) {
      acquire(ptr);
    }
    return *this;
  }

  /// Move assignment operator
  IntrusivePtr& operator=(IntrusivePtr &&other) {
    if (ptr) {
      release(ptr);
    }
    ptr = other.ptr;
    other.ptr = nullptr;
    return *this;
  }

  /// Destroy the intrusive ptr.
  virtual ~IntrusivePtr() {
    if (ptr) {
      release(ptr);
    }
  }

  /// Check whether the pointer is defined (ptr is not null).
  bool defined() const {return ptr != nullptr;}

  friend inline
  bool operator==(const IntrusivePtr<T> &p1, const IntrusivePtr<T> &p2) {
    return p1.ptr == p2.ptr;
  }

  friend inline
  bool operator!=(const IntrusivePtr<T> &p1, const IntrusivePtr<T> &p2) {
    return p1.ptr != p2.ptr;
  }

  friend inline
  bool operator<(const IntrusivePtr<T> &p1, const IntrusivePtr<T> &p2) {
    return p1.ptr < p2.ptr;
  }

  friend inline
  bool operator>(const IntrusivePtr<T> &p1, const IntrusivePtr<T> &p2) {
    return p1.ptr > p2.ptr;
  }

  friend inline
  bool operator<=(const IntrusivePtr<T> &p1, const IntrusivePtr<T> &p2) {
    return p1.ptr <= p2.ptr;
  }

  friend inline
  bool operator>=(const IntrusivePtr<T> &p1, const IntrusivePtr<T> &p2) {
    return p1.ptr >= p2.ptr;
  }
};

template <class Data>
class Manageable {
  friend void acquire(const Data *data) { ++data->ref; }
  friend void release(const Data *data) { if (--data->ref == 0) delete data; }

  mutable long ref = 0;
};

}} // namespace simit::util

#endif
