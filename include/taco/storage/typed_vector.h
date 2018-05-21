#ifndef TACO_STORAGE_TYPED_VECTOR_H
#define TACO_STORAGE_TYPED_VECTOR_H
#include <vector>
#include <taco/type.h>
#include <taco/storage/array.h>
#include <taco/storage/typed_value.h>
#include <taco/storage/typed_index.h>

namespace taco {
  namespace storage {
    // Like std::vector but for a dynamic DataType type. Backed by a char vector
    //templated over TypedComponentVal or TypedIndexVal
    template <typename Typed>
    class TypedVector {
    public:
      //based off of https://gist.github.com/jeetsukumaran/307264
      class iterator
      {
        public:
          typedef iterator self_type;
          typedef Typed value_type;
          typedef typename Typed::Ref reference;
          typedef typename Typed::Ptr pointer;
          typedef std::forward_iterator_tag iterator_category;
          typedef int difference_type;
          iterator(pointer ptr, DataType type) : ptr_(ptr), type(type) { }
          self_type operator++() { self_type i = *this; ptr_++; return i; }
          self_type operator++(int junk) { ptr_++; return *this; }
          reference operator*() { return *ptr_; }
          pointer operator->() { return ptr_; }
          bool operator==(const self_type& rhs) { return ptr_ == rhs.ptr_; }
          bool operator!=(const self_type& rhs) { return ptr_ != rhs.ptr_; }
        private:
          pointer ptr_;
          DataType type;
      };

      class const_iterator
      {
      public:
        typedef const_iterator self_type;
        typedef Typed value_type;
        typedef typename Typed::Ref reference;
        typedef typename Typed::Ptr pointer;
        typedef std::forward_iterator_tag iterator_category;
        typedef int difference_type;
        const_iterator(pointer ptr, DataType type) : ptr_(ptr), type(type) { }
        self_type operator++() { self_type i = *this; ptr_++; return i; }
        self_type operator++(int junk) { ptr_++; return *this; }
        reference operator*() { return *ptr_; }
        pointer operator->() { return ptr_; }
        bool operator==(const self_type& rhs) { return ptr_ == rhs.ptr_; }
        bool operator!=(const self_type& rhs) { return ptr_ != rhs.ptr_; }
      private:
        pointer ptr_;
        DataType type;
      };

      TypedVector() : type(DataType::Undefined) {
      }

      TypedVector(DataType type) : type(type) {
      }

      TypedVector(DataType type, size_t size) : type(type) {
        resize(size);
      }

      void push_back(void *value) {
        resize(size() + 1);
        set(size() - 1, value);
      }

      template<typename T>
      void push_back(T constant) {
        resize(size() + 1);
        set(size() - 1, constant);
      }
      void push_back(Typed value) {
        taco_iassert(value.getType() == type);
        resize(size() + 1);
        set(size() - 1, value);
      }

      void push_back(typename Typed::Ref value) {
        taco_iassert(value.getType() == type);
        resize(size() + 1);
        set(size() - 1, value);
      }

      void push_back_vector(TypedVector vector) {
        resize(size() + vector.size());
        memcpy(&get(size()-vector.size()).get(), vector.data(), type.getNumBytes()*vector.size());
      }

      template<typename T>
      void push_back_vector(std::vector<T> v) {
        resize(size() + v.size());
        for (size_t i = 0; i < v.size(); i++) {
          set(size() - v.size() + i, v.at(i));
        }
      }

      void resize(size_t size) {
        charVector.resize(size * type.getNumBytes());
      }

      typename Typed::Ref get(size_t index) const {
        return typename Typed::Ref(getType(), (void *) &charVector[index * type.getNumBytes()]);
      }

      void copyTo(size_t index, void *location) const;

      void set(size_t index, Typed value) {
        taco_iassert(value.getType() == type);
        get(index) = value;
      }
      void set(size_t index, typename Typed::Ref value) {
        taco_iassert(value.getType() == type);
        get(index) = value;
      }

      template<typename T>
      void set(size_t index, T constant) {
        set(index, Typed(type, constant));
      }

      template<typename T>
      void set(size_t index, T* value) {
        get(index) = typename Typed::Ref(type, (void *) value);
      }

      void clear() {
        charVector.clear();
      }

      size_t size() const {
        return charVector.size() / type.getNumBytes();
      }

      char* data() const {
        return (char *) charVector.data();
      }

      DataType getType() const {
        return type;
      }

      bool operator==(const TypedVector &other) const {
        if (size() != other.size()) return false;
        return (memcmp(data(), other.data(), size()*type.getNumBytes()) == 0);
      }

      bool operator!=(const TypedVector &other) const {
        return !(*this == other);
      }

        //needed to use in set
      bool operator>(const TypedVector &other) const {
        return !(*this < other) && !(*this == other);
      }

      bool operator<(const TypedVector &other) const {
        size_t minSize = size() < other.size() ? size() : other.size();
        for (size_t i = 0; i < minSize; i++) {
          if (get(i) < other.get(i)) return true;
          if (get(i) > other.get(i)) return false;
        }
        return size() < other.size();
      }

      iterator begin() {
        return iterator(&get(0), type);
      }

      iterator end() {
        return iterator(&get(size()), type);
      }

      const_iterator begin() const {
        return const_iterator(&get(0), type);
      }

      const_iterator end() const {
        return const_iterator(&get(size()), type);
      }

      typename Typed::Ref operator[] (const size_t index) const {
        return get(index);
      }

    private:
      std::vector<char> charVector;
      DataType type;
    };

    using TypedComponentVector = TypedVector<TypedComponentVal>;
    using TypedIndexVector = TypedVector<TypedIndexVal>;
  }
}
#endif
