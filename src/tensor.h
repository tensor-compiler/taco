#ifndef TACIT_TENSOR_H
#define TACIT_TENSOR_H

#include <vector>
#include <algorithm>
#include <memory>
#include <utility>
#include <iostream>

#include "format.h"
#include "expr.h"
#include "error.h"
#include "component_types.h"
#include "util/strings.h"
#include "util/variadic.h"
#include "util/comparable.h"
#include "util/intrusive_ptr.h"

namespace tacit {
class PackedTensor;
std::ostream& operator<<(std::ostream& os, const PackedTensor& tp);

struct Var;
struct Expr;

namespace internal {
class Stmt;
}

namespace util {
std::string uniqueName(char prefix);
}

template <typename T> struct Read;

template <typename T> class Tensor;

std::shared_ptr<PackedTensor>
pack(const std::vector<size_t>& dimensions, internal::ComponentType T,
     const Format& format, const std::vector<std::vector<int>>& coords,
     const void* values);

std::shared_ptr<internal::Stmt> lower(Expr expr);

template <typename T>
class TensorObject : public util::Manageable<TensorObject<T>> {
  friend class  Tensor<T>;
  friend struct Read<T>;

  struct Coordinate : util::Comparable<Coordinate> {
    template <typename... Indices>
    Coordinate(const std::vector<int>& loc, T val) : loc{loc}, val{val} {}

    std::vector<int> loc;
    T val;

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
        else if (l.loc[i] > r.loc[i]) return false;
      }
      return true;
    }
  };

  TensorObject(std::string name, std::vector<size_t> dimensions, Format format)
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

  const std::vector<Var>& getIndexVars() const {
    return indexVars;
  }

  Expr getExpr() const {
    return expr;
  }

  void insert(const std::vector<int>& coord, T val) {
    iassert(coord.size() == getOrder()) << "Wrong number of indices";
    coordinates.push_back(Coordinate(coord, val));
  }

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

    this->packedTensor = tacit::pack(dimensions, internal::typeOf<T>(),
                                     format, coords, values.data());
    coordinates.clear();
  }

  void compile() {
    iassert(expr.defined()) << "No expression defined for tensor";
    this->code = lower(expr);
  }

  void assemble() {
  }

  void evaluate() {
  }

  std::shared_ptr<PackedTensor> getPackedTensor() {
    return packedTensor;
  }

  const std::shared_ptr<PackedTensor> getPackedTensor() const {
    return packedTensor;
  }

  friend std::ostream& operator<<(std::ostream& os, const TensorObject<T>& t) {
    std::vector<std::string> dimStrings;
    for (int dim : t.getDimensions()) {
      dimStrings.push_back(std::to_string(dim));
    }
    os << t.getName()
       << " (" << util::join(dimStrings, "x") << ", " << t.format << ")";

    if (t.coordinates.size() > 0) {
      os << std::endl << "Coordinates: ";
      for (auto& coord : t.coordinates) {
        os << std::endl << "  (" << util::join(coord.loc) << "): " << coord.val;
      }
    }

    // Print packed data
    if (t.getPackedTensor() != nullptr) {
      os << std::endl << *t.getPackedTensor();
    }
    return os;
  }

  std::string                     name;
  std::vector<size_t>             dimensions;
  Format                          format;

  std::vector<Coordinate>         coordinates;
  std::shared_ptr<PackedTensor>   packedTensor;

  std::vector<Var>                indexVars;
  Expr                            expr;

  std::shared_ptr<internal::Stmt> code;
};

template <typename T>
class Tensor : public util::IntrusivePtr<TensorObject<T>> {
public:
  typedef size_t                  Dimension;
  typedef std::vector<Dimension>  Dimensions;
  typedef std::vector<int>        Coordinate;
  typedef std::pair<Coordinate,T> Value;

  Tensor(std::string name, Dimensions dimensions, Format format)
      : Tensor(new TensorObject(name, dimensions, format)) {
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
    getPtr()->insert(coord, val);
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
    getPtr()->pack();
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
    return os << *t.getPtr();
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
  typedef TensorObject<T> TensorObject;
  friend struct Read<T>;

  TensorObject* getPtr() const {
    return static_cast<TensorObject*>(util::IntrusivePtr<TensorObject>::ptr);
  }

  Read<T> operator()(const std::vector<Var>& indices) {
    return Read<T>(*this, indices);
  }

  Tensor(TensorObject* obj) : util::IntrusivePtr<TensorObject>(obj) {
  }
};

}

#endif
