#ifndef TACO_TENSOR_H
#define TACO_TENSOR_H

#include <cmath>
#include <vector>
#include <queue>
#include <algorithm>
#include <memory>
#include <utility>
#include <iostream>
#include <fstream>

#include "taco/tensor_base.h"
#include "taco/operator.h"
#include "taco/format.h"
#include "taco/expr.h"
#include "taco/component_types.h"
#include "storage/storage.h"
#include "taco/io/mtx_file_format.h"
#include "taco/util/error.h"
#include "taco/util/strings.h"
#include "taco/util/variadic.h"
#include "taco/util/comparable.h"
#include "taco/util/intrusive_ptr.h"

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
using namespace io;

template <typename C>
class Tensor : public TensorBase {
public:
//  typedef std::vector<int>        Dimensions;
  typedef std::vector<int>        Coordinate;
  typedef std::pair<Coordinate,C> Value;

  /// Create a scalar
  Tensor() : TensorBase() {}

  /// Create a scalar with the given name
  Tensor(std::string name) : Tensor(name, {}, Format()) {}

  /// Create a tensor with the given dimensions and format
  Tensor(std::vector<int> dimensions, Format format)
      : Tensor(util::uniqueName('A'), dimensions, format) {}

  /// Create a tensor with the given name, dimensions and format
  Tensor(std::string name, std::vector<int> dimensions,
         Format format, size_t allocSize = DEFAULT_ALLOC_SIZE)
      : TensorBase(name, typeOf<C>(), dimensions, format, allocSize) {
    taco_uassert(format.getLevels().size() == dimensions.size())
        << "The format size (" << format.getLevels().size()-1 << ") "
        << "of " << name
        << " does not match the dimension size (" << dimensions.size() << ")";
    taco_uassert(allocSize >= 2 && (allocSize & (allocSize - 1)) == 0)
        << "The initial index allocation size must be a power of two and "
        << "at least two";
  }

  void insert(const Coordinate& coord, C val) {
    taco_uassert(coord.size() == getOrder()) << "Wrong number of indices";
    taco_uassert(getComponentType() == typeOf<C>())
        << "Cannot insert a value of type '" << typeid(C).name() << "'";
    TensorBase::insert(coord, val);
  }

  void insert(const std::initializer_list<int>& coord, C val) {
    taco_uassert(coord.size() == getOrder()) << "Wrong number of indices";
    taco_uassert(getComponentType() == typeOf<C>())
        << "Cannot insert a value of type '" << typeid(C).name() << "'";
    TensorBase::insert(coord, val);
  }

  void insert(int coord, C val) {
    taco_uassert(1 == getOrder()) << "Wrong number of indices";
    taco_uassert(getComponentType() == typeOf<C>())
        << "Cannot insert a value of type '" << typeid(C).name() << "'";
    TensorBase::insert({coord}, val);
  }

  void insert(const Value& value) {
    insert(value.first, value.second);
  }

  void insertRow(int row_index, const std::vector<int>& col_index,
		 const std::vector<C>& values) {
    taco_iassert(col_index.size() == values.size());
    taco_iassert(getComponentType() == typeOf<C>());
    // TODO insert row by row method
    taco_not_supported_yet;
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

  Read operator()(const std::vector<Var>& indices) {
    taco_uassert(indices.size() == getOrder())
        << "A tensor of order " << getOrder() << " must be indexed with "
        << getOrder() << " variables. "
        << "Is indexed with: " << util::join(indices);
    return Read(*this, indices);
  }

  template <typename... Vars>
  Read operator()(const Vars&... indices) {
    taco_uassert(sizeof...(indices) == getOrder())
        << "A tensor of order " << getOrder() << " must be indexed with "
        << getOrder() << " variables. "
        << "Is indexed with: " << util::join(std::vector<Var>({indices...}));
    return Read(*this, {indices...});
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

   const_iterator operator++(int) {
     const_iterator result = *this;
     ++(*this);
     return result;
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
        count(1 + (size_t)isEnd * tensor->getStorage().getSize().numValues()),
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
          taco_not_supported_yet;
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
    taco_not_supported_yet;
  }

  Tensor<C>& operator+=(const Tensor<C>&) {
    taco_not_supported_yet;
  }

  Tensor<C>& operator-=(const Tensor<C>&) {
    taco_not_supported_yet;
  }

private:
  friend struct Read;
};


/// Tensor Negation
template <typename C>
Expr operator-(const Tensor<C>&) {
  taco_not_supported_yet;
  return Expr();
}

/// Tensor Scale
template <typename C>
Expr operator*(const Tensor<C>&, C) {
  taco_not_supported_yet;
  return Expr();
}

template <typename C>
Expr operator*(C, const Tensor<C>&) {
  taco_not_supported_yet;
  return Expr();
}

/// Tensor Addition
template <typename T>
Expr operator+(const Tensor<T>&, const Tensor<T>&) {
  taco_not_supported_yet;
  return Expr();
}

/// Tensor Subtraction
template <typename T>
Expr operator-(const Tensor<T>&, const Tensor<T>&) {
  taco_not_supported_yet;
  return Expr();
}

}
#endif
