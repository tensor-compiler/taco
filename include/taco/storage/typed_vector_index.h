#ifndef TACO_STORAGE_TYPED_INDEX_VECTOR_H
#define TACO_STORAGE_TYPED_INDEX_VECTOR_H
#include <vector>
#include <taco/type.h>
#include <taco/storage/array.h>
#include <taco/storage/typed_index.h>

namespace taco {
  namespace storage {
    // Like std::vector but for a dynamic DataType type. Backed by a char vector
    class TypedIndexVector {
    public:
      //based off of https://gist.github.com/jeetsukumaran/307264
      class iterator
      {
        public:
          typedef iterator self_type;
          typedef TypedIndexVal value_type;
          typedef TypedIndexRef reference;
          typedef TypedIndexPtr pointer;
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
          typedef TypedIndexVal value_type;
          typedef TypedIndexRef reference;
          typedef TypedIndexPtr pointer;
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

      TypedIndexVector();
      TypedIndexVector(DataType type);
      TypedIndexVector(DataType type, size_t size);
      void push_back(void *value);

      template<typename T>
      void push_back(T constant) {
        resize(size() + 1);
        set(size() - 1, constant);
      }
      void push_back(TypedIndexVal value);
      void push_back(TypedIndexRef value);
      void push_back_vector(TypedIndexVector vector);

      template<typename T>
      void push_back_vector(std::vector<T> v) {
        resize(size() + v.size());
        for (size_t i = 0; i < v.size(); i++) {
          set(size() - v.size() + i, v.at(i));
        }
      }
      void resize(size_t size);
      TypedIndexRef get(size_t index) const;
      void copyTo(size_t index, void *location) const;

      void set(size_t index, TypedIndexVal value);
      void set(size_t index, TypedIndexRef value);

      template<typename T>
      void set(size_t index, T constant) {
        set(index, TypedIndexVal(type, constant));
      }

      template<typename T>
      void set(size_t index, T* value) {
        get(index) = TypedIndexRef(type, (void *) value);
      }

      void clear();
      size_t size() const;
      char* data() const;
      DataType getType() const;
      bool operator==(const TypedIndexVector &other) const;
      bool operator!=(const TypedIndexVector &other) const;

      //needed to use in set
      bool operator>(const TypedIndexVector &other) const;
      bool operator<(const TypedIndexVector &other) const;

      iterator begin();
      iterator end();
      const_iterator begin() const;
      const_iterator end() const;
      
      TypedIndexRef operator[] (const size_t index) const;

    private:
      std::vector<char> charVector;
      DataType type;
    };
  }
}
#endif
