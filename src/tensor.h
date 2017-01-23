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
#include "storage/storage.h"
#include "util/strings.h"
#include "util/variadic.h"
#include "util/comparable.h"
#include "util/intrusive_ptr.h"
#include "util/fsm.h"

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

enum TensorState {
  UNDEFINED = -1,
  DEFINED,
  COMPILED,
  ASSEMBLED,
  COMPUTED
};

template <typename C>
class Tensor {
public:
  typedef std::vector<int>        Dimensions;
  typedef std::vector<int>        Coordinate;
  typedef std::pair<Coordinate,C> Value;

  Tensor(std::string name, Dimensions dimensions, 
         Format format, size_t allocSize = 1 << 20)
      : tensor(internal::Tensor(name, dimensions, format, 
                                internal::typeOf<C>(), allocSize)) {
    uassert(format.getLevels().size() == dimensions.size())
        << "The format size (" << format.getLevels().size()-1 << ") "
        << "of " << name
        << " does not match the dimension size (" << dimensions.size() << ")";
    uassert(allocSize >= 2 && (allocSize & (allocSize - 1)) == 0)
        << "The initial index allocation size must be a power of two and "
        << "at least two";
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
  const Format& getFormat() const {
    return tensor.getFormat();
  }

  void insert(const Coordinate& coord, C val) {
    uassert(coord.size() == getOrder()) << "Wrong number of indices";
    uassert(tensor.getComponentType() == internal::typeOf<C>()) 
        << "Cannot insert a value of type '" << typeid(C).name() << "'";
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
    tensor.compile();
  }

  // Assemble the tensor storage, including index and value arrays.
  void assemble() {
    tensor.assemble();
  }

  // evaluate the values into the tensor storage.
  void compute() {
    tensor.compute();
  }

  void eval() {
    compile();
    assemble();
    compute();
  }

  friend std::ostream& operator<<(std::ostream& os, const Tensor<C>& t) {
    return os << t.tensor;
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

    const_iterator(const Tensor<C>* tensor, bool isEnd = false) : 
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
      const auto& levels = tensor->getFormat().getLevels();

      if (lvl == tensor->getOrder()) {
        if (advance) {
          advance = false;
          return false;
        }

        const size_t idx = (lvl == 0) ? 0 : ptrs[lvl - 1];
        curVal.second = tensor->getStorage().getValues()[idx];

        for (size_t i = 0; i < lvl; ++i) {
          const size_t dim = levels[i].getDimension();
          curVal.first[dim] = coord[i];
        }

        advance = true;
        return true;
      }
      
      const auto storage    = tensor->getStorage();
      const auto levelIndex = storage.getLevelIndex(lvl);

      switch (levels[lvl].getType()) {
        case Dense: {
          const auto dim  = levelIndex.ptr[0];
          const auto base = (lvl == 0) ? 0 : (ptrs[lvl - 1] * dim);

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
          const auto& segs = levelIndex.ptr;
          const auto& vals = levelIndex.idx;
          const auto  k    = (lvl == 0) ? 0 : ptrs[lvl - 1];

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
        case Fixed: {
          const auto  elems = levelIndex.ptr[0];
          const auto  base  = (lvl == 0) ? 0 : (ptrs[lvl - 1] * elems);
          const auto& vals  = levelIndex.idx;

          if (advance) {
            goto resume_fixed;
          }

          for (ptrs[lvl] = base; 
               ptrs[lvl] < base + elems && vals[ptrs[lvl]] >= 0; ++ptrs[lvl]) {
            coord[lvl] = vals[ptrs[lvl]];

          resume_fixed:
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

    const Tensor<C>*  tensor;
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

  Tensor<C>& operator*=(C) {
    not_supported_yet;
  }

  Tensor<C>& operator+=(const Tensor<C>&) {
    not_supported_yet;
  }

  Tensor<C>& operator-=(const Tensor<C>&) {
    not_supported_yet;
  }

private:
  friend struct Read;
  internal::Tensor tensor;
};


/// Tensor Negation
template <typename C>
Expr operator-(const Tensor<C>&) {
  not_supported_yet;
  return Expr();
}

/// Tensor Scale
template <typename C>
Expr operator*(const Tensor<C>&, C) {
  not_supported_yet;
  return Expr();
}

template <typename C>
Expr operator*(C, const Tensor<C>&) {
  not_supported_yet;
  return Expr();
}

/// Tensor Addition
template <typename T>
Expr operator+(const Tensor<T>&, const Tensor<T>&) {
  not_supported_yet;
  return Expr();
}

/// Tensor Subtraction
template <typename T>
Expr operator-(const Tensor<T>&, const Tensor<T>&) {
  not_supported_yet;
  return Expr();
}

}
#endif
