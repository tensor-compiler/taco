#ifndef TACO_TENSOR_H
#define TACO_TENSOR_H

#include <vector>
#include <algorithm>
#include <memory>
#include <utility>
#include <iostream>

#include "format.h"
#include "expr.h"
#include "error.h"
#include "component_types.h"
#include "ir_printer.h"
#include "util/strings.h"
#include "util/variadic.h"
#include "util/comparable.h"
#include "util/intrusive_ptr.h"

namespace taco {
class PackedTensor;
std::ostream& operator<<(std::ostream& os, const PackedTensor& tp);

struct Var;
struct Expr;

namespace util {
std::string uniqueName(char prefix);
}

template <typename T> struct Read;
template <typename T> class Tensor;

namespace internal {
class Stmt;

class Tensor : public util::Manageable<Tensor> {
  friend class  taco::Tensor<double>;
  friend struct Read<double>;

  Tensor(std::string name, std::vector<size_t> dimensions, Format format)
      : name(name), dimensions(dimensions), format(format) {
  }

  std::string getName() const {
    return name;
  }

  Format getFormat() const {
    return format;
  }

  const std::vector<size_t>& getDimensions() const {
    return dimensions;
  }

  size_t getOrder() const {
    return dimensions.size();
  }

  const std::vector<taco::Var>& getIndexVars() const {
    return indexVars;
  }

  taco::Expr getExpr() const {
    return expr;
  }

  void pack(const std::vector<std::vector<int>>& coords,
            internal::ComponentType ctype, const void* values);

  void compile();
  void assemble();
  void evaluate();

  std::shared_ptr<PackedTensor> getPackedTensor() {
    return packedTensor;
  }

  const std::shared_ptr<PackedTensor> getPackedTensor() const {
    return packedTensor;
  }

  friend std::ostream& operator<<(std::ostream& os, const internal::Tensor& t) {
    std::vector<std::string> dimStrings;
    for (int dim : t.getDimensions()) {
      dimStrings.push_back(std::to_string(dim));
    }
    os << t.getName()
       << " (" << util::join(dimStrings, "x") << ", " << t.format << ")";

    // Print packed data
    if (t.getPackedTensor() != nullptr) {
      os << std::endl << *t.getPackedTensor();
    }
    return os;
  }

  std::string                     name;
  std::vector<size_t>             dimensions;
  Format                          format;

  std::shared_ptr<PackedTensor>   packedTensor;

  std::vector<taco::Var>          indexVars;
  taco::Expr                      expr;

  std::shared_ptr<internal::Stmt> code;
};
} // namespace internal

template <typename T>
class Tensor : public util::IntrusivePtr<internal::Tensor> {
public:
  typedef size_t                  Dimension;
  typedef std::vector<Dimension>  Dimensions;
  typedef std::vector<int>        Coordinate;
  typedef std::pair<Coordinate,T> Value;

  Tensor(std::string name, Dimensions dimensions, Format format)
      : Tensor(new internal::Tensor(name, dimensions, format)) {
  }

  Tensor(Dimensions dimensions, Format format)
      : Tensor(util::uniqueName('A'), dimensions, format) {
  }

  Tensor(std::string name, Dimensions dimensions, std::string format)
      : Tensor(name, dimensions, Format(format)) {
  }

  Tensor(Dimensions dimensions, std::string format)
      : Tensor(util::uniqueName('A'), dimensions, format) {
  }

  std::string getName() const {
    return getPtr()->getName();
  }

  const std::vector<size_t>& getDimensions() const {
    return getPtr()->getDimensions();
  }

  size_t getOrder() {
    return getPtr()->getOrder();
  }

  /// Get the format the tensor is packed into
  Format getFormat() const {
    return getPtr()->getFormat();
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
    std::sort(coordinates.begin(), coordinates.end());

    // convert coords to structure of arrays
    std::vector<std::vector<int>> coords(getOrder());
    for (size_t i=0; i < getOrder(); ++i) {
      coords[i] = std::vector<int>(coordinates.size());
    }

    std::vector<T> values(coordinates.size());
    for (size_t i=0; i < coordinates.size(); ++i) {
      for (size_t d=0; d < getOrder(); ++d) {
        coords[d][i] = coordinates[i].loc[d];
      }
      values[i] = coordinates[i].val;
    }

    getPtr()->pack(coords, internal::typeOf<T>(), values.data());

    coordinates.clear();
  }

  template <typename... Vars>
  Read<T> operator()(const Vars&... indices) {
    return operator()({indices...});
  }

  /// Compile the tensor expression.
  void compile() {
    uassert(getExpr().defined())
        << "The tensor does not have an expression to evaluate";
    getPtr()->compile();
  }

  // Assemble the tensor storage, including index and value arrays.
  void assemble() {
    // TODO: assert tensor has been compiled
    getPtr()->assemble();
  }

  // evaluate the values into the tensor storage.
  void evaluate() {
    // TODO: assert tensor has been compiled
    // TODO: assert tensor has been assembled
    getPtr()->evaluate();
  }

  friend std::ostream& operator<<(std::ostream& os, const Tensor<T>& t) {
    os << *t.getPtr();
    if (t.coordinates.size() > 0) {
      os << std::endl << "Coordinates: ";
      for (auto& coord : t.coordinates) {
        os << std::endl << "  (" << util::join(coord.loc) << "): " << coord.val;
      }
    }
    return os;
  }

  const std::vector<Var>& getIndexVars() const {
    return getPtr()->getIndexVars();
  }

  template <typename E = Expr> E getExpr() const {
    return to<E>(getPtr()->getExpr());
  }

  const std::shared_ptr<PackedTensor> getPackedTensor() const {
    return getPtr()->getPackedTensor();
  }

private:
  friend struct Read<T>;

  struct Coord : util::Comparable<Coordinate> {
    template <typename... Indices>
    Coord(const std::vector<int>& loc, T val) : loc{loc}, val{val} {}

    std::vector<int> loc;
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

  internal::Tensor* getPtr() const {
    return static_cast<internal::Tensor*>(util::IntrusivePtr<internal::Tensor>::ptr);
  }

  Read<T> operator()(const std::vector<Var>& indices) {
    return Read<T>(*this, indices);
  }

  Tensor(internal::Tensor* obj) : util::IntrusivePtr<internal::Tensor>(obj) {
  }
};

}

#endif
