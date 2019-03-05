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
#include "taco/error/error_messages.h"

namespace taco {

/// Structure to hold a non zero as a tuple (coordinate, value).
///
/// CType the type of the value stored.
/// Order the number of dimensions of the component
template<size_t Order, typename CType>
class Component {
public:
  Component() : coord(), val(0) {}

  Component(Coordinate<Order> coordinate, CType value) : coord(coordinate), val(value) {
    taco_uassert(coord.order() == Order) <<
      "Wrong number of indices";
  }

  size_t coordinate(int mode) const {
    taco_uassert(mode < Order) << "requested mode coordinate exceeds order of component.";
    return coord[mode];
  }

  const Coordinate<Order> coordinate() const {
    return coord;
  }

  const CType& value() const { return val; }

private:
  Coordinate<Order> coord;
  CType val;
};

}
#endif