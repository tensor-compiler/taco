#ifndef TAC_TENSOR_H
#define TAC_TENSOR_H

#include <vector>
#include <tuple>
#include <array>
#include <algorithm>
#include <iostream>

#include "format.h"
#include "expr.h"
#include "error.h"
#include "scalar_types.h"
#include "util/strings.h"
#include "util/variadic.h"
#include "util/comparable.h"

namespace tac {

template <typename CType, int... dims>
class Tensor {
public:
  Tensor(Format format) : format(format) {}

  Format getFormat() const {return format;}
  static constexpr size_t getOrder() {return sizeof...(dims);}

  void insert(const std::vector<int>& indices, CType val) {
    coordinates.push_back(Coordinate(indices, val));
  }

  void pack() {
    std::sort(coordinates.begin(), coordinates.end());
    
  }

  const std::vector<std::vector<int>>& getIndices() {return indices;}
  const std::vector<std::vector<int>>& getValues() {return indices;}

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
    if (t.values.size() > 0) {
      os << std::endl;
      os << "print packed data";
    }
    return os;
  }

private:
  Format format;

  std::vector<std::vector<int>> indices;
  std::vector<CType> values;

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
