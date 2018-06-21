#ifndef TACO_UTIL_COMPARABLE_H
#define TACO_UTIL_COMPARABLE_H

namespace taco {
namespace util {

/// Interface for classes that can be compared to each other. Classes that
/// implement this interfaces define `==` and `<` to get `!=`, `>`, `<=`, and
/// `>=` for free.
template <class T>
class Comparable {};

template <class T>
bool operator!=(const Comparable<T> &lhs, const Comparable<T> &rhs) {
  return !(*static_cast<const T*>(&lhs) == *static_cast<const T*>(&rhs));
}

template <class T>
bool operator>(const Comparable<T> &lhs, const Comparable<T> &rhs) {
  return !(*static_cast<const T*>(&lhs) < *static_cast<const T*>(&rhs) ||
           *static_cast<const T*>(&lhs) == *static_cast<const T*>(&rhs));
}

template <class T>
bool operator<=(const Comparable<T> &lhs, const Comparable<T> &rhs) {
  return !(*static_cast<const T*>(&lhs) > *static_cast<const T*>(&rhs));
}

template <class T>
bool operator>=(const Comparable<T> &lhs, const Comparable<T> &rhs) {
  return !(*static_cast<const T*>(&lhs) < *static_cast<const T*>(&rhs));
}

}}
#endif
