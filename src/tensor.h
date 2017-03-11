#ifndef TACO_TENSOR_H
#define TACO_TENSOR_H

#include <vector>
#include <queue>
#include <algorithm>
#include <memory>
#include <utility>
#include <iostream>
#include <fstream>

#include "tensor_base.h"
#include "operator.h"
#include "format.h"
#include "expr.h"
#include "component_types.h"
#include "storage/storage.h"
#include "io/hb_file_format.h"
#include "io/mtx_file_format.h"

#include "util/error.h"
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
    uassert(format.getLevels().size() == dimensions.size())
        << "The format size (" << format.getLevels().size()-1 << ") "
        << "of " << name
        << " does not match the dimension size (" << dimensions.size() << ")";
    uassert(allocSize >= 2 && (allocSize & (allocSize - 1)) == 0)
        << "The initial index allocation size must be a power of two and "
        << "at least two";
  }

  void insert(const Coordinate& coord, C val) {
    uassert(coord.size() == getOrder()) << "Wrong number of indices";
    uassert(getComponentType() == typeOf<C>())
        << "Cannot insert a value of type '" << typeid(C).name() << "'";
    TensorBase::insert(coord, val);
  }

  void insert(const std::initializer_list<int>& coord, C val) {
    uassert(coord.size() == getOrder()) << "Wrong number of indices";
    uassert(getComponentType() == typeOf<C>())
        << "Cannot insert a value of type '" << typeid(C).name() << "'";
    TensorBase::insert(coord, val);
  }

  void insert(int coord, C val) {
    uassert(1 == getOrder()) << "Wrong number of indices";
    uassert(getComponentType() == typeOf<C>())
        << "Cannot insert a value of type '" << typeid(C).name() << "'";
    TensorBase::insert({coord}, val);
  }

  void insert(const Value& value) {
    insert(value.first, value.second);
  }

  // Write a sparse matrix to a file stored in the MTX format.
  void writeMTX(std::string MTXfilename) const {
    uassert(getFormat().isCSC()) <<
        "writeMTX: the tensor " << getName() <<
        " is not defined in the CSC format";
    std::ofstream MTXfile;

    MTXfile.open(MTXfilename.c_str());
    uassert(MTXfile.is_open())
            << " Error opening the file " << MTXfilename.c_str();

    auto S = getStorage();
    auto size = S.getSize();

    int nrow = getDimensions()[0];
    int ncol = getDimensions()[1];
    int nnzero = size.values;
    std::string name = getName();

    mtx::writeFile(MTXfile, name,
                   nrow,ncol,nnzero);

    for (const auto& val : *this) {
      MTXfile << val.first[0]+1 << " " << val.first[1]+1 << " " ;
      if (std::floor(val.second) == val.second)
        MTXfile << val.second << ".0 " << std::endl;
      else
        MTXfile << val.second << " " << std::endl;
    }

    MTXfile.close();
  }

  void insertRow(int row_index, const std::vector<int>& col_index,
		 const std::vector<C>& values) {
    iassert(col_index.size() == values.size());
    iassert(getComponentType() == typeOf<C>());
    // TODO insert row by row method
    not_supported_yet;
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
    uassert(indices.size() == getOrder())
        << "A tensor of order " << getOrder() << " must be indexed with "
        << getOrder() << " variables. "
        << "Is indexed with: " << util::join(indices);
    return Read(*this, indices);
  }

  template <typename... Vars>
  Read operator()(const Vars&... indices) {
    uassert(sizeof...(indices) == getOrder())
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
