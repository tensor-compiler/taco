#ifndef TACO_COMPONENT_H
#define TACO_COMPONENT_H

#include <initializer_list>
#include <algorithm>
#include <array>
#include <ostream>
#include "taco/coordinate.h"
#include "taco/util/comparable.h"
#include "taco/util/strings.h"
#include "taco/error.h"

namespace taco {

/// Structure to hold a non zero as a tuple (coordinate, value).
///
/// CType the type of the value stored.
/// Order the number of dimensions of the component
template<size_t Order, typename CType>
class Component {
public:
  Component() : coordinate(), value(0) {}

  Component(Coordinate<Order> coordinate, CType v) : coordinate(coordinate), value(v) {
    taco_uassert(coordinate.order() == Order) <<
      "Wrong number of indices";
  }

  size_t coordinate(int mode) const {
    taco_uassert(mode < Order) << "requested mode coordinate exceeds order of component.";
    return coordinate[mode];
  }

  const Coordinate<Order> coordinate() const {
    return coordinate;
  }

  const CType& value() const { return value; }

private:
  Coordinate<Order> coordinate;
  T value;
};

#endif