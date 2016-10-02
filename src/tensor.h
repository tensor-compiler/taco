#ifndef TACO_TENSOR_H
#define TACO_TENSOR_H

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
#include "util/intrusive_ptr.h"

namespace taco {
class PackedTensor;

struct Var;
struct Expr;

template <typename CType> struct Read;

template <typename CType> class Tensor;

std::shared_ptr<PackedTensor>
pack(const std::vector<size_t>& dimensions, internal::ComponentType ctype,
     const Format& format, size_t ncoords, const void* coords,
     const void* values);

template <typename CType>
class TensorObject : public util::Manageable<TensorObject<CType>> {
  friend class  Tensor<CType>;
  friend struct Read<CType>;

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

  TensorObject(const std::vector<size_t>& dimensions, Format format)
      : dimensions(dimensions), format(format) {}

  Format getFormat() const { return format; }

  const std::vector<size_t>& getDimensions() const { return dimensions; }

  size_t getOrder() const { return dimensions.size(); }

  const std::vector<Var>& getIndices() const { return indices; }

  Expr getSource() const { return source; }

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

    this->packedTensor = taco::pack(dimensions, internal::typeOf<CType>(),
                                    format, coordinates.size(),
                                    coords.data(), values.data());
    coordinates.clear();
  }

  std::shared_ptr<PackedTensor> getPackedTensor() {
    return packedTensor;
  }

  const std::shared_ptr<PackedTensor> getPackedTensor() const {
    return packedTensor;
  }

  friend std::ostream& operator<<(std::ostream& os, 
                                  const TensorObject<CType>& t) {
    std::vector<std::string> dimensions;
    for (int dim : t.dimensions) {
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
//      os << std::endl;
    }
    return os;
  }

  std::vector<size_t>           dimensions;
  Format                        format;
  std::shared_ptr<PackedTensor> packedTensor;
  std::vector<Coordinate>       coordinates;
  std::vector<Var>              indices;
  Expr                          source;
};

template <typename CType>
class Tensor : public util::IntrusivePtr<TensorObject<CType>> {
  friend struct Read<CType>;

  typedef TensorObject<CType> TensorObject;

  TensorObject* getPtr() const {
    return static_cast<TensorObject*>(util::IntrusivePtr<TensorObject>::ptr);
  }

  Read<CType> operator()(const std::vector<Var>& indices) {
    return Read<CType>(*this, indices);
  }

public:
  Tensor(TensorObject* obj) : util::IntrusivePtr<TensorObject>(obj) {}
  Tensor(const std::vector<size_t>& dimensions, Format format) : 
      Tensor(new TensorObject(dimensions, format)) {}

  Format getFormat() const { return getPtr()->getFormat(); }

  const std::vector<size_t>& getDimensions() const {
    return getPtr()->getDimensions();
  }

  size_t getOrder() { return getPtr()->getOrder(); }

  const std::vector<Var>& getIndices() const { return getPtr()->getIndices(); }

  template <typename E = Expr>
  E getSource() const { return to<E>(getPtr()->getSource()); }

  void insert(const std::vector<int>& coord, CType val) {
    getPtr()->insert(coord, val);
  }

  void pack() { getPtr()->pack(); }

  std::shared_ptr<PackedTensor> getPackedTensor() { 
    return getPtr()->getPackedTensor(); 
  }

  const std::shared_ptr<PackedTensor> getPackedTensor() const {
    return getPtr()->getPackedTensor();
  }
  
  template <typename... Vars>
  Read<CType> operator()(const Vars&... indices) {
    return operator()({indices...});
  }

  friend std::ostream& operator<<(std::ostream& os, const Tensor<CType>& t) {
    return os << *t.getPtr();
  }
};

}

#endif
