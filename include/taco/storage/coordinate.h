#ifndef TACO_STORAGE_COORDINATE_H
#define TACO_STORAGE_COORDINATE_H

#include <initializer_list>
#include <algorithm>
#include <array>
#include <ostream>
#include "taco/util/comparable.h"
#include "taco/util/strings.h"
#include "taco/error.h"

namespace taco {

template <size_t Order, typename Type=int64_t>
class Coordinates : util::Comparable<Coordinates<Order, Type>> {
public:
  template <typename... T>
  Coordinates(T... coordinates) : coordinates{{coordinates...}} {}

  Type& operator[](size_t idx) {
    taco_iassert(idx < Order);
    return coordinates[idx];
  }

  const Type& operator[](size_t idx) const {
    taco_iassert(idx < Order);
    return coordinates[idx];
  }

  template <size_t O, typename T>
  friend bool operator==(const Coordinates<O,T>& a, const Coordinates<O,T>& b) {
    for (size_t i = 0; i < Order; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

  template <size_t O, typename T>
  friend bool operator<(const Coordinates<O,T>& a, const Coordinates<O,T>& b) {
    for (size_t i = 0; i < Order; i++) {
      if (a[i] < b[i]) return true;
      if (a[i] > b[i]) return false;
    }
    return false;
  }

  template <size_t O, typename T>
  friend std::ostream& operator<<(std::ostream& os, const Coordinates<O,T>& c) {
    return os << util::join(c.coordinates);
  }

  template <size_t O, typename T, typename V>
  friend std::ostream& operator<<(std::ostream& os,
                                  const std::pair<Coordinates<O,T>, V>& c) {
    return os << "(" << util::join(c.first.coordinates, ",") << "):" << c.second;
  }

private:
  std::array<Type,Order> coordinates;
};

}
#endif
