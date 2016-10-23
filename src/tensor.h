#ifndef TACO_TENSOR_H
#define TACO_TENSOR_H

#include <vector>
#include <algorithm>
#include <memory>
#include <utility>
#include <iostream>

#include "internal_tensor.h"
#include "operator.h"
#include "format.h"
#include "expr.h"
#include "error.h"
#include "component_types.h"
#include "util/strings.h"
#include "util/variadic.h"
#include "util/comparable.h"
#include "util/intrusive_ptr.h"

namespace taco {
class PackedTensor;
std::ostream& operator<<(std::ostream& os, const PackedTensor& tp);

class Var;
class Expr;

namespace util {
std::string uniqueName(char prefix);
}

struct Read;

namespace internal {
class Stmt;
}

template <typename T>
class Tensor {
public:
  typedef size_t                  Dimension;
  typedef std::vector<Dimension>  Dimensions;
  typedef std::vector<uint32_t>   Coordinate;
  typedef std::pair<Coordinate,T> Value;

  Tensor(std::string name, Dimensions dimensions, Format format)
      : tensor(internal::Tensor(name, dimensions, format)) {
    uassert(format.getLevels().size() == dimensions.size())
        << "The format size (" << format.getLevels().size()-1 << ") "
        << "of " << name
        << " does not match the dimension size (" << dimensions.size() << ")";
  }

  Tensor(Dimensions dimensions, Format format)
      : Tensor(util::uniqueName('A'), dimensions, format) {
  }

  Tensor(Dimensions dimensions, std::string format)
      : Tensor(util::uniqueName('A'), dimensions, format) {
  }

  std::string getName() const {
    return tensor.getName();
  }

  const std::vector<size_t>& getDimensions() const {
    return tensor.getDimensions();
  }

  size_t getOrder() {
    return tensor.getOrder();
  }

  /// Get the format the tensor is packed into
  Format getFormat() const {
    return tensor.getFormat();
  }

  void insert(const Coordinate& coord, T val) {
    iassert(coord.size() == getOrder()) << "Wrong number of indices";
    coordinates.push_back(Coord(coord, val));
  }

  void insert(const Value& value) {
    insert(value.first, value.second);
  }

  void insert(const std::vector<Value>& values) {
    for (auto& value : values) {
      insert(value);
    }
  }

  /// Pack tensor into the given format
  void pack() {
    iassert(getFormat().getLevels().size() == getOrder());

    // Packing code currently only packs coordinates in the order of the
    // dimensions. To work around this we just permute each coordinate according
    // to the storage dimensions.
    auto levels = getFormat().getLevels();
    std::vector<size_t> permutation;
    for (auto& level : levels) {
      permutation.push_back(level.getDimension());
    }

    std::vector<Coord> permutedCoords;
    permutation.reserve(coordinates.size());
    for (size_t i=0; i < coordinates.size(); ++i) {
      auto& coord = coordinates[i];
      std::vector<uint32_t> ploc(coord.loc.size());
      for (size_t j=0; j < getOrder(); ++j) {
        ploc[permutation[j]] = coord.loc[j];
      }
      permutedCoords.push_back(Coord(ploc, coord.val));
    }
    coordinates.clear();

    // The pack code requires the coordinates to be sorted
    std::sort(permutedCoords.begin(), permutedCoords.end());

    // convert coords to structure of arrays
    std::vector<std::vector<int>> coords(getOrder());
    for (size_t i=0; i < getOrder(); ++i) {
      coords[i] = std::vector<int>(permutedCoords.size());
    }

    std::vector<T> values(permutedCoords.size());
    for (size_t i=0; i < permutedCoords.size(); ++i) {
      for (size_t d=0; d < getOrder(); ++d) {
        coords[d][i] = permutedCoords[i].loc[d];
      }
      values[i] = permutedCoords[i].val;
    }

    tensor.pack(coords, internal::typeOf<T>(), values.data());
  }

  Read operator()(const std::vector<Var>& indices) {
    uassert(indices.size() == getOrder())
        << "A tensor of order " << getOrder() << " must be indexed with "
        << getOrder() << " variables. "
        << "Is indexed with: " << util::join(indices);
    return Read(tensor, indices);
  }

  template <typename... Vars>
  Read operator()(const Vars&... indices) {
    uassert(sizeof...(indices) == getOrder())
        << "A tensor of order " << getOrder() << " must be indexed with "
        << getOrder() << " variables. "
        << "Is indexed with: " << util::join(std::vector<Var>({indices...}));
    return Read(tensor, {indices...});
  }

  /// Compile the tensor expression.
  void compile() {
    uassert(getExpr().defined())
        << "The tensor does not have an expression to evaluate";
    tensor.compile();
  }

  // Assemble the tensor storage, including index and value arrays.
  void assemble() {
    // TODO: assert tensor has been compiled
    tensor.assemble();
  }

  // evaluate the values into the tensor storage.
  void evaluate() {
    // TODO: assert tensor has been compiled
    // TODO: assert tensor has been assembled
    tensor.evaluate();
  }

  friend std::ostream& operator<<(std::ostream& os, const Tensor<T>& t) {
    os << t.tensor;
    if (t.coordinates.size() > 0) {
      os << std::endl << "Coordinates: ";
      for (auto& coord : t.coordinates) {
        os << std::endl << "  (" << util::join(coord.loc) << "): " << coord.val;
      }
    }
    return os;
  }

  const std::vector<Var>& getIndexVars() const {
    return tensor.getIndexVars();
  }

  template <typename E = Expr>
  E getExpr() const {
    return to<E>(tensor.getExpr());
  }

  const std::shared_ptr<PackedTensor> getPackedTensor() const {
    return tensor.getPackedTensor();
  }

  void printIterationSpace() const {
    tensor.printIterationSpace();
  }

private:
  friend struct Read;

  struct Coord : util::Comparable<Coordinate> {
    template <typename... Indices>
    Coord(const std::vector<uint32_t>& loc, T val) : loc{loc}, val{val} {}

    std::vector<uint32_t> loc;
    T val;

    friend bool operator==(const Coord& l, const Coord& r) {
      iassert(l.loc.size() == r.loc.size());
      for (size_t i=0; i < l.loc.size(); ++i) {
        if (l.loc[i] != r.loc[i]) return false;
      }
      return true;
    }
    friend bool operator<(const Coord& l, const Coord& r) {
      iassert(l.loc.size() == r.loc.size());
      for (size_t i=0; i < l.loc.size(); ++i) {
        if (l.loc[i] < r.loc[i]) return true;
        else if (l.loc[i] > r.loc[i]) return false;
      }
      return true;
    }
  };

  std::vector<Coord> coordinates;
  internal::Tensor   tensor;
};

}

#endif
