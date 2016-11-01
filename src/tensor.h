#ifndef TACO_TENSOR_H
#define TACO_TENSOR_H

#include <vector>
#include <queue>
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
class Var;
class Expr;
struct Read;

namespace storage {
class Storage;
}

namespace ir {
class Stmt;
}

namespace util {
std::string uniqueName(char prefix);
}


template <typename T>
class Tensor {
public:
  typedef std::vector<int>        Dimensions;
  typedef std::vector<int>        Coordinate;
  typedef std::pair<Coordinate,T> Value;

  Tensor(std::string name, Dimensions dimensions, Format format) : tensor(
      internal::Tensor(name, dimensions, format, internal::typeOf<T>())) {
    uassert(format.getLevels().size() == dimensions.size())
        << "The format size (" << format.getLevels().size()-1 << ") "
        << "of " << name
        << " does not match the dimension size (" << dimensions.size() << ")";
  }

  Tensor(Dimensions dimensions, Format format)
      : Tensor(util::uniqueName('A'), dimensions, format) {
  }

  std::string getName() const {
    return tensor.getName();
  }

  const std::vector<int>& getDimensions() const {
    return tensor.getDimensions();
  }

  size_t getOrder() const {
    return tensor.getOrder();
  }

  /// Get the format the tensor is packed into
  Format getFormat() const {
    return tensor.getFormat();
  }

  void insert(const Coordinate& coord, T val) {
    uassert(coord.size() == getOrder()) << "Wrong number of indices";
    uassert(tensor.getComponentType() == internal::typeOf<T>()) 
        << "Cannot insert a value of type '" << typeid(T).name() << "'";
    tensor.insert(coord, val);
  }

  void insert(const Value& value) {
    insert(value.first, value.second);
  }

  template <class InputIterator>
  void insert(const InputIterator begin, const InputIterator end) {
    for (InputIterator it = begin; it != end; ++it) {
      insert(*it);
    }
  }

  void insert(const std::vector<Value>& values) {
    insert(values.begin(), values.end());
  }

  /// Pack tensor into the given format
  void pack() {
    tensor.pack();
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
  void compute() {
    // TODO: assert tensor has been compiled
    // TODO: assert tensor has been assembled
    tensor.compute();
  }

  friend std::ostream& operator<<(std::ostream& os, const Tensor<T>& t) {
    os << t.tensor;
    return os;
  }

  const std::vector<Var>& getIndexVars() const {
    return tensor.getIndexVars();
  }

  template <typename E = Expr>
  E getExpr() const {
    return to<E>(tensor.getExpr());
  }

  const storage::Storage& getStorage() const {
    return tensor.getStorage();
  }

  void printIterationSpace() const {
    tensor.printIterationSpace();
  }

  void printIR(std::ostream& os) const {
    tensor.printIR(os);
  }

  class const_iterator {
  public:
    typedef const_iterator self_type;
    typedef Value value_type;
    typedef Value& reference;
    typedef Value* pointer;
    typedef std::forward_iterator_tag iterator_category;

    const_iterator(const const_iterator&) = default;

    const_iterator operator++() {
      advanceIndex();
      return *this;
    }

    const Value& operator*() const {
      return curVal;
    }

    const Value* operator->() const {
      return &curVal;
    }

    bool operator==(const const_iterator& rhs) {
      return tensor == rhs.tensor && count == rhs.count;
    }

    bool operator!=(const const_iterator& rhs) {
      return !(*this == rhs);
    }

  private:
    friend class Tensor;

    const_iterator(const Tensor<T>* tensor, bool isEnd = false) : 
        tensor(tensor),
        coord(Coordinate(tensor->getOrder())),
        ptrs(Coordinate(tensor->getOrder())),
        curVal(Value(Coordinate(tensor->getOrder()), 0)),
        count(1 + (size_t)isEnd * tensor->getStorage().getSize().values),
        advance(false) {
      advanceIndex();
    }

    void advanceIndex() {
      advanceIndex(0);
      ++count;
    }

    bool advanceIndex(size_t lvl) {
      const auto& levels  = tensor->getFormat().getLevels();
      auto storage    = tensor->getStorage();
      auto levelIndex = storage.getLevelIndex(lvl);

      if (lvl == tensor->getOrder()) {
        if (advance) {
          advance = false;
          return false;
        }

        curVal.second = tensor->getStorage().getValues()[ptrs[lvl - 1]];

        for (size_t i = 0; i < lvl; ++i) {
          const size_t dim = levels[i].getDimension();
          curVal.first[dim] = coord[i];
        }

        advance = true;
        return true;
      }

      switch (levels[lvl].getType()) {
        case Dense: {
          auto dim = levelIndex.ptr[0];
          const int base = (lvl == 0) ? 0 : (ptrs[lvl - 1] * dim);

          if (advance) {
            goto resume_dense;  // obligatory xkcd: https://xkcd.com/292/
          }

          for (coord[lvl] = 0; coord[lvl] < dim; ++coord[lvl]) {
            ptrs[lvl] = base + coord[lvl];

          resume_dense:
            if (advanceIndex(lvl + 1)) {
              return true;
            }
          }
          break;
        }
        case Sparse: {
          const auto&  segs = levelIndex.ptr;
          const auto&  vals = levelIndex.idx;
          const size_t k    = (lvl == 0) ? 0 : ptrs[lvl - 1];

          if (advance) {
            goto resume_sparse;
          }

          for (ptrs[lvl] = segs[k]; ptrs[lvl] < segs[k + 1]; ++ptrs[lvl]) {
            coord[lvl] = vals[ptrs[lvl]];

          resume_sparse:
            if (advanceIndex(lvl + 1)) {
              return true;
            }
          }
          break;
        }
        default:
          not_supported_yet;
          break;
      }

      return false;
    }

    const Tensor<T>*  tensor;
    Coordinate        coord;
    Coordinate        ptrs;
    Value             curVal;
    size_t            count;
    bool              advance;
  };

  const_iterator begin() const {
    return const_iterator(this);
  }

  const_iterator end() const {
    return const_iterator(this, true);
  }

private:
  friend struct Read;

  internal::Tensor tensor;
};

}

#endif
