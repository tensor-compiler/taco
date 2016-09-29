#ifndef TAC_TENSOR_H
#define TAC_TENSOR_H

#include <vector>
#include <algorithm>
#include <memory>
#include <iostream>

#include "format.h"
#include "expr.h"
#include "error.h"
#include "component_types.h"
#include "util/strings.h"
#include "util/variadic.h"
#include "util/comparable.h"

namespace tac {
struct PackedTensor;

std::shared_ptr<PackedTensor>
pack(const std::vector<int>& dimensions, internal::ComponentType ctype,
     const Format& format, size_t ncoords, const void* coords,
     const void* values);

template <typename CType, int... dims>
class Tensor {
public:
  Tensor(Format format) : format(format) {}

  Format getFormat() const {return format;}
  static constexpr size_t getOrder() {return sizeof...(dims);}

  void insert(const std::vector<int>& coord, CType val) {
    iassert(coord.size() == getOrder()) << "Wrong number of indices";
    coordinates.push_back(Coordinate(coord, val));
  }

  void pack() {
    std::sort(coordinates.begin(), coordinates.end());

    std::vector<int>   coords(coordinates.size() * getOrder());
    std::vector<CType> values(coordinates.size());
    for (size_t i=0; i < coordinates.size(); ++i) {
      for (size_t d=0; d < getOrder(); ++d) {
        coords[i*getOrder() + d] = coordinates[i].loc[d];
      }
      values[i] = coordinates[i].val;
    }

    this->packedTensor = tac::pack({dims...}, internal::typeOf<CType>(),
                                   format, coordinates.size(),
                                   coords.data(), values.data());
  }

  std::shared_ptr<PackedTensor> getPackedTensor() {
    return packedTensor;
  }

  const std::shared_ptr<PackedTensor> getPackedTensor() const {
    return packedTensor;
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const Tensor<CType,dims...>& t) {
    std::vector<std::string> dimensions;
    for (int dim : {dims...}) {
      dimensions.push_back(std::to_string(dim));
    }
    os << util::join(dimensions, "x") << "-tensor (" << t.format << ")";

    if (t.coordinates.size() > 0) {
      for (auto& coord : t.coordinates) {
        os << std::endl << "  s(" << util::join(coord.loc) << "): " <<coord.val;
      }
    }

    // Print packed data
    if (t.getPackedTensor() != nullptr) {
      os << std::endl;
      os << "print packed data";
    }
    return os;
  }

private:
  Format format;
  std::shared_ptr<PackedTensor> packedTensor;

  struct Coordinate : util::Comparable<Coordinate> {
    template <typename... Indices>
    Coordinate(const std::vector<int>& loc, CType val) : loc{loc}, val{val} {}

    std::vector<int> loc;
    CType val;

    friend bool operator==(const Coordinate& l, const Coordinate& r) {
      iassert(l.loc.size() == r.loc.size());
      for (size_t i=0; i < l.loc.size(); ++i) {
        if (l.loc[i] != r.loc[i]) return false;
      }
      return true;
    }
    friend bool operator<(const Coordinate& l, const Coordinate& r) {
      iassert(l.loc.size() == r.loc.size());
      for (size_t i=0; i < l.loc.size(); ++i) {
        if (l.loc[i] < r.loc[i]) return true;
        else if (l.loc[i] < r.loc[i]) return false;
      }
      return true;
    }
  };
  std::vector<Coordinate> coordinates;
};

}
#endif
