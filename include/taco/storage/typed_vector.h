#ifndef TACO_STORAGE_TYPED_VECTOR_H
#define TACO_STORAGE_TYPED_VECTOR_H
#include <cstring>
#include <vector>
#include <taco/type.h>
#include <taco/storage/array.h>
#include <taco/storage/typed_value.h>
#include <taco/storage/typed_index.h>

namespace taco {

/// Like std::vector but for a dynamic DataType type. Backed by a char vector
/// Templated over TypedComponentVal or TypedIndexVal
template <typename Typed>
class TypedVector {
public:
  /// Create a new empty TypedVector of undefined type
  TypedVector() : type(Datatype::Undefined) {
  }

  /// Create a new empty TypedVector of specified type
  TypedVector(Datatype type) : type(type) {
  }

  /// Create a new empty TypedVector initialized to specified size
  TypedVector(Datatype type, size_t size) : type(type) {
    resize(size);
  }

  /// Accesses the value at a given index
  typename Typed::Ref operator[] (const size_t index) const {
    return get(index);
  }

  /// Accesses the value at a given index
  typename Typed::Ref get(size_t index) const {
    return typename Typed::Ref(getType(), (void *) &charVector[index * type.getNumBytes()]);
  }

  /// Gets the type of the TypedVector
  Datatype getType() const {
    return type;
  }

  /// Gets the size of the TypedVector
  size_t size() const {
    return charVector.size() / type.getNumBytes();
  }

  /// Resizes the TypedVector to a given size (in number of items)
  void resize(size_t size) {
    charVector.resize(size * type.getNumBytes());
  }

  /// Returns a pointer to the underlying char vector
  char* data() const {
    return (char *) charVector.data();
  }

  /// Clears the data from the TypedVector
  void clear() {
    charVector.clear();
  }

  /// Sets the value at a given index to either TypedComponentVal or TypedIndexVal
  void set(size_t index, Typed value) {
    taco_iassert(value.getType() == type);
    get(index) = value;
  }

  /// Sets the value at a given index to either TypedComponentRef or TypedIndexRef
  void set(size_t index, typename Typed::Ref value) {
    taco_iassert(value.getType() == type);
    get(index) = value;
  }

  /// Sets the value at a given index to a constant cast to the type of the TypedVector
  template<typename T>
  void set(size_t index, T constant) {
    set(index, Typed(type, constant));
  }

  /// Sets the value at a given index to the value stored at a pointer of the type of the TypedVector
  template<typename T>
  void set(size_t index, T* value) {
    get(index) = typename Typed::Ref(type, (void *) value);
  }


  /// Push value at pointer of type of the TypedVector
  void push_back(void *value) {
    resize(size() + 1);
    set(size() - 1, value);
  }

  /// Push constant casted to type of the TypedVector
  template<typename T>
  void push_back(T constant) {
    resize(size() + 1);
    set(size() - 1, constant);
  }

  /// Push typed value (either TypedComponentVal or TypedIndexVal)
  void push_back(Typed value) {
    taco_iassert(value.getType() == type);
    resize(size() + 1);
    set(size() - 1, value);
  }

  /// Push typed reference (either TypedComponentRef or TypedIndexRef)
  void push_back(typename Typed::Ref value) {
    taco_iassert(value.getType() == type);
    resize(size() + 1);
    set(size() - 1, value);
  }

  /// Push back all values in TypedVector
  void push_back_vector(TypedVector vector) {
    taco_iassert(type == vector.getType());
    resize(size() + vector.size());
    memcpy(&get(size()-vector.size()).get(), vector.data(), type.getNumBytes()*vector.size());
  }

  /// Push back all values in std::vector
  template<typename T>
  void push_back_vector(std::vector<T> v) {
    resize(size() + v.size());
    for (size_t i = 0; i < v.size(); i++) {
      set(size() - v.size() + i, v.at(i));
    }
  }


  /// Compare two TypedVectors
  bool operator==(const TypedVector &other) const {
    if (size() != other.size()) return false;
    if (type != other.getType()) return false;
    return (memcmp(data(), other.data(), size()*type.getNumBytes()) == 0);
  }

  /// Compare two TypedVectors
  bool operator!=(const TypedVector &other) const {
    return !(*this == other);
  }

  /// Comparison operator needed to use in Set
  bool operator>(const TypedVector &other) const {
    return !(*this < other) && !(*this == other);
  }

  /// Comparison operator needed to use in Set. Uses lexicographical comparison.
  bool operator<(const TypedVector &other) const {
    size_t minSize = size() < other.size() ? size() : other.size();
    for (size_t i = 0; i < minSize; i++) {
      if (get(i) < other.get(i)) return true;
      if (get(i) > other.get(i)) return false;
    }
    return size() < other.size();
  }


  /// Iterator for vector
  /// Implementation based off of https://gist.github.com/jeetsukumaran/307264
  class iterator
  {
  public:
    /// Typedef for self
    typedef iterator self_type;
    /// Typedef for type of value (either TypedComponentVal or TypedIndexVal)
    typedef Typed value_type;
    /// Type of reference of value_type (either TypedComponentRef or TypedIndexRef)
    typedef typename Typed::Ref reference;
    /// Type of pointer of value_type (either TypedComponentPtr or TypedIndexPtr)
    typedef typename Typed::Ptr pointer;
    /// Help compiler choose right iterator
    typedef std::forward_iterator_tag iterator_category;

    /// Create an iterator from pointer and datatype
    iterator(pointer ptr, Datatype type) : ptr_(ptr), type(type) { }

    /// Post-increment iterator
    self_type operator++(int junk) { self_type i = *this; ptr_++; return i; }
    /// Pre-increment iterator
    self_type operator++() { ptr_++; return *this; }
    /// Get current reference of iterator
    reference operator*() { return *ptr_; }
    /// Get current pointer of iterator
    pointer operator->() { return ptr_; }

    /// Equality comparison of two iterators
    bool operator==(const self_type& rhs) { return ptr_ == rhs.ptr_; }
    /// Inequality comparison of two iterators
    bool operator!=(const self_type& rhs) { return ptr_ != rhs.ptr_; }
  private:
    /// Current pointer of iterator
    pointer ptr_;
    /// Data type of iterator
    Datatype type;
  };

  /// Constant iterator for vector
  /// Implementation based off of https://gist.github.com/jeetsukumaran/307264
  class const_iterator
  {
  public:
    /// Typedef for self
    typedef const_iterator self_type;
    /// Typedef for type of value (either TypedComponentVal or TypedIndexVal)
    typedef Typed value_type;
    /// Type of reference of value_type (either TypedComponentRef or TypedIndexRef)
    typedef typename Typed::Ref reference;
    /// Type of pointer of value_type (either TypedComponentPtr or TypedIndexPtr)
    typedef typename Typed::Ptr pointer;
    /// Help compiler choose right iterator
    typedef std::forward_iterator_tag iterator_category;

    /// Create an iterator from pointer and datatype
    const_iterator(pointer ptr, Datatype type) : ptr_(ptr), type(type) { }

    /// Post-increment iterator
    self_type operator++(int junk) { self_type i = *this; ptr_++; return i; }
    /// Pre-increment iterator
    self_type operator++() { ptr_++; return *this; }
    /// Get current reference of iterator
    reference operator*() { return *ptr_; }
    /// Get current pointer of iterator
    pointer operator->() { return ptr_; }

    /// Equality comparison of two iterators
    bool operator==(const self_type& rhs) { return ptr_ == rhs.ptr_; }
    /// Inequality comparison of two iterators
    bool operator!=(const self_type& rhs) { return ptr_ != rhs.ptr_; }
  private:
    /// Current pointer of iterator
    pointer ptr_;
    /// Data type of iterator
    Datatype type;
  };

  /// Return iterator at start of TypedVector
  iterator begin() {
    return iterator(&get(0), type);
  }

  /// Return iterator at end of TypedVector
  iterator end() {
    return iterator(&get(size()), type);
  }

  /// Return constant iterator at start of TypedVector
  const_iterator begin() const {
    return const_iterator(&get(0), type);
  }

  /// Return constant iterator at end of TypedVector
  const_iterator end() const {
    return const_iterator(&get(size()), type);
  }

private:
  /// Char vector that backs the TypedVector
  std::vector<char> charVector;
  /// Type of items in TypedVector
  Datatype type;
};

/// Type alias for a TypedVector templated to TypedComponentVals
using TypedComponentVector = TypedVector<TypedComponentVal>;

/// Type alias for a TypedVector templated to TypedIndexVals
using TypedIndexVector = TypedVector<TypedIndexVal>;

}
#endif
