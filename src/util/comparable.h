#ifndef TACIT_UTIL_COMPARABLE_H
#define TACIT_UTIL_COMPARABLE_H

namespace tacit {
namespace util {

/// Interface for classes that can be compared to each other. Classes that
/// implement this interfaces must define == and <, to get !=, >, <=, >=.
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
