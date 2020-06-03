#ifndef TACO_COORDINATE_H
#define TACO_COORDINATE_H

#include <initializer_list>
#include <algorithm>
#include <array>
#include <ostream>
#include "taco/util/comparable.h"
#include "taco/util/strings.h"
#include "taco/error.h"

namespace taco {

/// Structure to represent a multidimensional coordinate tuple (i,j,k,...).
template <size_t Order, typename Type=int64_t>
class Coordinate : util::Comparable<Coordinate<Order, Type>> {
public:
  Coordinate() {}

  template <typename... T>
  Coordinate(T... coordinates) : coordinates{{coordinates...}} {}

  Type& operator[](size_t idx) {
    taco_iassert(idx < Order);
    return coordinates[idx];
  }

  const Type& operator[](size_t idx) const {
    taco_iassert(idx < Order);
    return coordinates[idx];
  }

  operator std::vector<int>() const {
    std::vector<int> vec;
    for (auto coord : coordinates) {
      vec.push_back(coord);
    }
    return vec;
  }

  size_t order() {
    return Order;
  }

  // friend methods

  template <size_t O, typename T>
  friend bool operator==(const Coordinate<O,T>& a, const Coordinate<O,T>& b) {
    for (size_t i = 0; i < Order; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

  template <size_t O, typename T>
  friend bool operator<(const Coordinate<O,T>& a, const Coordinate<O,T>& b) {
    for (size_t i = 0; i < Order; i++) {
      if (a[i] < b[i]) return true;
      if (a[i] > b[i]) return false;
    }
    return false;
  }

  template <size_t O, typename T>
  friend std::ostream& operator<<(std::ostream& os, const Coordinate<O,T>& c) {
    return os << util::join(c.coordinates);
  }

  template <size_t O, typename T, typename V>
  friend std::ostream& operator<<(std::ostream& os,
                                  const std::pair<Coordinate<O,T>, V>& c) {
    return os << "(" << util::join(c.first.coordinates, ",") << "):" << c.second;
  }

private:
  std::array<Type,Order> coordinates;
};

}
#endif