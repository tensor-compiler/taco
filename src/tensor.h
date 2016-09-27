#ifndef TAC_TENSOR_H
#define TAC_TENSOR_H

#include <vector>
#include <tuple>
#include <array>
#include <iostream>

#include "format.h"
#include "expr.h"
#include "util/strings.h"
#include "util/variadic.h"

namespace tac {

template <typename CType, int... dims>
class Tensor {
public:
  Tensor(Format format) : format(format) {}

  Format getFormat() const {return format;}
  static constexpr size_t getOrder() {return sizeof...(dims);}

  void insert(int index, CType val) {
    static_assert(getOrder() == 1, "Wrong number of indices.");
    coordinates.push_back(Coordinate({index}, val));
  }

  void insert(const std::vector<int>& indices, CType val) {
    coordinates.push_back(Coordinate(indices, val));
  }

  void pack() {
    // TODO
  }

  const std::vector<std::vector<int>>& getIndices() {return indices;}
  const std::vector<std::vector<int>>& getValues() {return indices;}

  friend std::ostream& operator<<(std::ostream& os,
                                  const Tensor<CType,dims...>& t){
    // TODO
    return os;
  }

private:
  Format format;
  std::vector<std::vector<int>> indices;
  std::vector<CType> values;

  struct Coordinate {
    template <typename... Indices>
    Coordinate(const std::vector<int>& loc, CType val) : loc{loc}, val{val} {}

    std::vector<int> loc;
    CType val;
  };
  std::vector<Coordinate> coordinates;
};


}
#endif
