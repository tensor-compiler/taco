#ifndef TACO_UTIL_COLLECTIONS_H
#define TACO_UTIL_COLLECTIONS_H

#include <map>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <functional>
#include <vector>
#include <taco/cuda.h>

namespace taco {
namespace util {

/// Query whether a collection contains an element
template <class C, typename V>
bool contains(const C &container, const V &value) {
  return std::find(container.begin(),container.end(),value) != container.end();
}

/// Query whether a set contains an element
template <typename V>
bool contains(const std::set<V> &container, const V &value) {
  return container.find(value) != container.end();
}

/// Query whether a map contains an element
template <typename K, typename V>
bool contains(const std::map<K,V> &container, const K &key) {
  return container.find(key) != container.end();
}

/// Query whether a set contains an element
template <typename V>
bool contains(const std::unordered_set<V> &container, const V &value) {
  return container.find(value) != container.end();
}

/// Query whether a map contains an element
template <typename K, typename V>
bool contains(const std::unordered_map<K,V> &container, const K &key) {
  return container.find(key) != container.end();
}

/// Append all values of a collection to a vector
template <typename V, class C>
void append(std::vector<V>& vector, const C& container) {
  vector.insert(std::end(vector), std::begin(container), std::end(container));
}

template <typename V>
void append(std::vector<V>& vector, const std::initializer_list<V>& container) {
  append(vector, std::vector<V>(container));
}

/// Prepend all values of a collection to a vector
template <typename V, class C>
void prepend(std::vector<V>& vector, const C& container) {
  vector.insert(vector.begin(), container.begin(), container.end());
}

template <typename V>
void prepend(std::vector<V>& vector, const std::initializer_list<V>& container){
  prepend(vector, std::vector<V>(container));
}

template <typename V>
std::vector<V> combine(const std::vector<V>& a, const std::vector<V>& b) {
  std::vector<V> result;
  append(result, a);
  append(result, b);
  return result;
}

template <typename V>
std::vector<V> combine(const std::vector<V>& a, const std::vector<V>& b,
                       const std::vector<V>& c) {
  std::vector<V> result;
  append(result, a);
  append(result, b);
  append(result, c);
  return result;
}

template <typename V>
std::vector<V> remove(const std::vector<V>& vector, 
                      const std::vector<V>& filter) {
  std::vector<V> result;
  for (const V& v : vector) {
    if (!contains(filter, v)) {
      result.push_back(v);
    }
  }
  return result;
}

template <typename V, typename T>
std::vector<V> filter(const std::vector<V>& vector, T test) {
  std::vector<V> result;
  for (auto& element : vector) {
    if (test(element)) {
      result.push_back(element);
    }
  }
  return result;
}

template <typename V, typename T>
size_t count(const std::vector<V>& vector, T test) {
  size_t count = 0;
  for (auto& element : vector) {
    if (test(element)) {
      count++;
    }
  }
  return count;
}

/**
 * Split the vector into two vectors where elements in the first pass the test
 * and elements in the second do not.
 */
template <typename V, typename T>
std::pair<std::vector<V>,std::vector<V>> split(const std::vector<V>& vector,
                                               T test) {
  std::vector<V> first;
  std::vector<V> second;
  for (auto& element : vector) {
    if (test(element)) {
      first.push_back(element);
    }
    else {
      second.push_back(element);
    }
  }
  return {first, second};
}

template <typename C, typename T>
bool all(const C& collection, T test) {
  for (const auto& element : collection) {
    if (!test(element)) {
      return false;
    }
  }
  return true;
}

template <typename V, typename T>
bool any(const std::vector<V>& vector, T test) {
  for (auto& element : vector) {
    if (test(element)) {
      return true;
    }
  }
  return false;
}

/// Copy vector to an array.
template <typename T>
T* copyToArray(const std::vector<T>& vec) {
  size_t size = vec.size() * sizeof(T);
  T* array;
  if (should_use_CUDA_unified_memory()) {
    array = static_cast<T*>(cuda_unified_alloc(size));
  }
  else {
    array = static_cast<T*>(malloc(size));
  }
  memcpy(array, vec.data(), size);
  return array;
}

template <typename T>
T* copyToArray(const std::initializer_list<T>& initList) {
  return copyToArray(std::vector<T>(initList));
}

template <typename T>
std::vector<T> copyToVector(T* ptr, size_t size) {
  std::vector<T> vec(size);
  for (size_t i = 0; i < size; ++i) {
    vec[i] = ptr[i];
  }
  return vec;
}

template <typename T>
std::set<T> toSet(const std::vector<T>& collection) {
  return std::set<T>(collection.begin(), collection.end());
}

/// Retrieve the location in the collection of the given value
template <class Collection, typename Value>
size_t locate(const Collection &collection, const Value &value) {
  assert(util::contains(collection, value));
  return std::distance(collection.begin(), std::find(collection.begin(),
                                                     collection.end(), value));
}

/// Map the elements of a collection to a vector, using the function f.
template <class Collection, typename ParamValue, typename ResultValue>
std::vector<ResultValue> map(const Collection &collection,
                             std::function<ResultValue(ParamValue)> f) {
  std::vector<ResultValue> result;
  result.resize(collection.size());
  std::transform(collection.begin(), collection.end(), result.begin(), f);
  return result;
}

// Iterables
template <class Collection>
class ReverseIterable {
public:
  typedef typename Collection::reverse_iterator reverse_iterator;
  ReverseIterable(Collection &c) : c(c) {}
  reverse_iterator begin() {return c.rbegin();}
  reverse_iterator end() {return c.rend();}
private:
  Collection &c;
};

template <class Collection>
class ReverseConstIterable {
public:
  typedef typename Collection::const_reverse_iterator const_reverse_iterator;
  ReverseConstIterable(const Collection &c) : c(c) {}
  const_reverse_iterator begin() const {return c.rbegin();}
  const_reverse_iterator end() const {return c.rend();}
private:
  const Collection &c;
};

template <class Collection>
class ExcludeFirstIterable {
public:
  typedef typename Collection::iterator iterator;
  ExcludeFirstIterable(Collection &c) : c(c) {}
  iterator begin() {return (c.begin() == c.end() ? c.end() : ++c.begin());}
  iterator end() {return c.end();}
private:
  Collection &c;
};

template <class Collection>
class ExcludeFirstConstIterable {
public:
  typedef typename Collection::const_iterator const_iterator;
  ExcludeFirstConstIterable(const Collection &c) : c(c) {}
  const_iterator begin() const {return c.begin()==c.end()?c.end():++c.begin();}
  const_iterator end() const {return c.end();}
private:
  const Collection &c;
};


template <class C1, class C2>
class ZipConstIterable {
public:
  typedef typename C1::value_type value_type1;
  typedef typename C2::value_type value_type2;
  friend std::ostream& operator<<(std::ostream &os,
                                  const std::pair<value_type1,value_type2> &p) {
    return os << "(" << p.first << ", " << p.second << ")";
  }
  class ZipConstIterator {
  public:
    typedef typename C1::const_iterator const_iterator1;
    typedef typename C2::const_iterator const_iterator2;
    ZipConstIterator(const_iterator1 c1it, const_iterator2 c2it)
        : c1it(c1it), c2it(c2it) {}
    ZipConstIterator(const ZipConstIterator& zit)
        : c1it(zit.c1it), c2it(zit.c2it) {}
    ZipConstIterator& operator++() {
      ++c1it;
      ++c2it;
      return *this;
    }
    ZipConstIterator operator++(int) {
      ZipConstIterator tmp(*this);
      operator++();
      return tmp;
    }
    friend bool operator==(const ZipConstIterator &l,
                           const ZipConstIterator &r) {
      return l.c1it == r.c1it && l.c2it == r.c2it;
    }
    friend bool operator!=(const ZipConstIterator &l,
                           const ZipConstIterator &r) {
      return l.c1it != r.c1it || l.c2it != r.c2it;
    }
    std::pair<value_type1, value_type2> operator*() {
      return std::pair<value_type1, value_type2>(*c1it, *c2it);
    }

  private:
    const_iterator1 c1it;
    const_iterator2 c2it;
  };
  ZipConstIterable(const C1 &c1, const C2 &c2) : c1(c1), c2(c2) {}
  ZipConstIterator begin() const {
    return ZipConstIterator(c1.begin(),c2.begin());
  }
  ZipConstIterator end() const {
    return ZipConstIterator(c1.end(), c2.end());
  }
private:
  const C1 &c1;
  const C2 &c2;
};

/// Iterate over a collection in reverse using a range for loop:
/// for (auto &element : util::reverse(collection)) {...}
template <class C>
ReverseIterable<C> reverse(C &collection) {
  return ReverseIterable<C>(collection);
}

/// Iterate over a collection in reverse using a range for loop:
/// for (auto &element : util::reverse(collection)) {...}
template <class C>
ReverseConstIterable<C> reverse(const C &collection) {
  return ReverseConstIterable<C>(collection);
}

/// Iterate over the elements in a collection, excluding the first element.
template <class C>
ExcludeFirstIterable<C> excludeFirst(C &collection) {
  return ExcludeFirstIterable<C>(collection);
}

/// Iterate over the elements in a collection, excluding the first element.
template <class C>
ExcludeFirstConstIterable<C> excludeFirst(const C &collection) {
  return ExcludeFirstConstIterable<C>(collection);
}

/// Zip the iterators of two collections. If one collection is smaller, the zip
/// iterates util there are no more elements in that collection.
template <class C1, class C2>
ZipConstIterable<C1, C2> zip(const C1 &collection1, const C2 &collection2) {
  return ZipConstIterable<C1, C2>(collection1, collection2);
}

}}
#endif
