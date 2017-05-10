#ifndef TACO_TENSOR_H
#define TACO_TENSOR_H

#include <string>
#include <vector>

#include "taco/tensor_base.h"
#include "taco/format.h"
#include "storage/storage.h"
#include "taco/util/error.h"

namespace taco {
namespace ir {
class Stmt;
}
namespace util {
std::string uniqueName(char prefix);
}

template <typename CType>
class Tensor : public TensorBase {
public:
  typedef std::vector<int>            Coordinate;
  typedef std::pair<Coordinate,CType> Value;

  /// Create a scalar
  Tensor() : TensorBase() {}

  /// Create a scalar with the given name
  Tensor(std::string name) : Tensor(name, {}, Format()) {}

  /// Create a tensor with the given dimensions and format
  Tensor(std::vector<int> dimensions, Format format)
      : Tensor(util::uniqueName('A'), dimensions, format) {}

  /// Create a tensor with the given name, dimensions and format
  Tensor(std::string name, std::vector<int> dimensions, Format format)
      : TensorBase(name, type<CType>(), dimensions, format) {
    taco_uassert(format.getLevels().size() == dimensions.size())
        << "The format size (" << format.getLevels().size()-1 << ") "
        << "of " << name
        << " does not match the dimension size (" << dimensions.size() << ")";
  }

  /// Create a tensor from a TensorBase instance. The Tensor and TensorBase
  /// objects will reference the same underlying tensor so it is a shallow copy.
  Tensor(const TensorBase& tensor) : TensorBase(tensor) {
    taco_uassert(tensor.getComponentType() == type<CType>()) <<
        "Assigning TensorBase with " << tensor.getComponentType() <<
        " components to a Tensor<" << type<CType>() << ">";
  }

  void insert(const Coordinate& coord, CType val) {
    taco_uassert(coord.size() == getOrder()) << "Wrong number of indices";
    taco_uassert(getComponentType() == type<CType>())
        << "Cannot insert a value of type '" << typeid(CType).name() << "'";
    TensorBase::insert(coord, val);
  }

  void insert(const std::initializer_list<int>& coord, CType val) {
    taco_uassert(coord.size() == getOrder()) << "Wrong number of indices";
    taco_uassert(getComponentType() == type<CType>())
        << "Cannot insert a value of type '" << typeid(CType).name() << "'";
    TensorBase::insert(coord, val);
  }

  void insert(int coord, CType val) {
    taco_uassert(1 == getOrder()) << "Wrong number of indices";
    taco_uassert(getComponentType() == type<CType>())
        << "Cannot insert a value of type '" << typeid(CType).name() << "'";
    TensorBase::insert({coord}, val);
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

    const_iterator(const Tensor<CType>* tensor, bool isEnd = false) : 
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
      
      const auto storage = tensor->getStorage();
      const auto index = storage.getDimensionIndex(lvl);

      switch (levels[lvl].getType()) {
        case Dense: {
          const auto dim  = index[0][0];
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
          const auto& segs = index[0];
          const auto& vals = index[1];
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
          const auto  elems = index[0][0];
          const auto  base  = (lvl == 0) ? 0 : (ptrs[lvl - 1] * elems);
          const auto& vals  = index[1];

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

    const Tensor<CType>* tensor;
    Coordinate           coord;
    Coordinate           ptrs;
    Value                curVal;
    size_t               count;
    bool                 advance;
  };

  const_iterator begin() const {
    return const_iterator(this);
  }

  const_iterator end() const {
    return const_iterator(this, true);
  }
};


/// Iterate over the typed values of a TensorBase.
template <typename CType>
Tensor<CType> iterate(const TensorBase& tensor) {
  return Tensor<CType>(tensor);
}

}
#endif
