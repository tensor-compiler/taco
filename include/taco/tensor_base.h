#ifndef TACO_INTERNAL_TENSOR_H
#define TACO_INTERNAL_TENSOR_H

#include <memory>
#include <string>
#include <vector>

#include <iostream>

#include "taco/format.h"
#include "taco/component_types.h"
#include "taco/util/comparable.h"
#include "taco/util/strings.h"
#include "storage/storage.h"

namespace taco {
class Var;
class Expr;

namespace storage {
class Storage;
}

const size_t DEFAULT_ALLOC_SIZE = (1 << 20);

class TensorBase : public util::Comparable<TensorBase> {
public:
  /// Create an undefined tensor
  TensorBase();

  /// Create a scalar
  TensorBase(ComponentType ctype);

  /// Create a scalar with the given name
  TensorBase(std::string name, ComponentType ctype);

  /// Create a tensor with the given dimensions and format
  TensorBase(ComponentType ctype, std::vector<int> dimensions, Format format,
             size_t allocSize = DEFAULT_ALLOC_SIZE);

  /// Create a tensor with the given dimensions and format
  TensorBase(std::string name, ComponentType ctype,
             std::vector<int> dimensions, Format format,
             size_t allocSize = DEFAULT_ALLOC_SIZE);

  std::string getName() const;
  size_t getOrder() const;
  const std::vector<int>& getDimensions() const;

  /// Get the format the tensor is packed into
  const Format& getFormat() const;

  const ComponentType& getComponentType() const;
  const std::vector<taco::Var>& getIndexVars() const;
  const taco::Expr& getExpr() const;
  const storage::Storage& getStorage() const;
  size_t getAllocSize() const;

  void insert(const std::vector<int>& coord, int val);
  void insert(const std::vector<int>& coord, float val);
  void insert(const std::vector<int>& coord, double val);
  void insert(const std::vector<int>& coord, bool val);

  void setCSR(double* vals, int* rowPtr, int* colIdx);
  void getCSR(double** vals, int** rowPtr, int** colIdx);

  void setCSC(double* vals, int* colPtr, int* rowIdx);
  void getCSC(double** vals, int** colPtr, int** rowIdx);

  /// Read a tensor from a file. The file type is determined from the file name
  /// extension.
  void read(std::string filename);

  /// Read a sparse matrix from a file stored in the Harwell-Boeing format.
  void readHB(std::string filename);

  /// Write a sparse matrix to a file stored in the Harwell-Boeing format.
  void writeHB(std::string filename) const;

  /// Read a sparse matrix from a file stored in the MTX format.
  void readMTX(std::string filename, int blockSize);

  /// Write a sparse matrix to a file stored in the MTX format.
  void writeMTX(std::string filename) const;

  /// Pack tensor into the given format
  void pack();

  /// Compile the tensor expression.
  void compile();

  /// Assemble the tensor storage, including index and value arrays.
  void assemble();

  /// Zero out the values
  void zero();

  /// Compute the given expression and put the values in the tensor storage.
  void compute();

  /// Compile, assemble and compute as needed.
  void evaluate();

  void setExpr(taco::Expr expr);
  void setIndexVars(std::vector<taco::Var> indexVars);

  void printIR(std::ostream&) const;
  void printComputeIR(std::ostream&, bool color) const;
  void printAssemblyIR(std::ostream&, bool color) const;

  friend bool operator!=(const TensorBase&, const TensorBase&);
  friend bool operator<(const TensorBase&, const TensorBase&);

  struct Coordinate : util::Comparable<Coordinate> {
    typedef std::vector<int> Coord;

    Coordinate(const Coord& loc, int    val) : loc(loc), ival(val) {}
    Coordinate(const Coord& loc, float  val) : loc(loc), fval(val) {}
    Coordinate(const Coord& loc, double val) : loc(loc), dval(val) {}
    Coordinate(const Coord& loc, bool   val) : loc(loc), bval(val) {}

    Coord loc;
    union {
      int    ival;
      float  fval;
      double dval;
      bool   bval;
    };

    friend bool operator==(const Coordinate& l, const Coordinate& r) {
      taco_iassert(l.loc.size() == r.loc.size());
      return l.loc == r.loc;
    }
    friend bool operator<(const Coordinate& l, const Coordinate& r) {
      taco_iassert(l.loc.size() == r.loc.size());
      return l.loc < r.loc;
    }
  };
  
  class const_iterator {
  public:
    typedef const_iterator self_type;
    typedef Coordinate value_type;
    typedef Coordinate& reference;
    typedef Coordinate* pointer;
    typedef std::forward_iterator_tag iterator_category;

    const_iterator(const const_iterator&) = default;

    const_iterator operator++() {
      advanceIndex();
      return *this;
    }

    const Coordinate& operator*() const {
      return curVal;
    }

    const Coordinate* operator->() const {
      return &curVal;
    }

    bool operator==(const const_iterator& rhs) {
      return tensor == rhs.tensor && count == rhs.count;
    }

    bool operator!=(const const_iterator& rhs) {
      return !(*this == rhs);
    }

  private:
    friend class TensorBase;

    const_iterator(const TensorBase* tensor, bool isEnd = false) : 
        tensor(tensor),
        coord(Coordinate::Coord(tensor->getOrder())),
        ptrs(Coordinate::Coord(tensor->getOrder())),
        curVal(Coordinate(Coordinate::Coord(tensor->getOrder()), 0)),
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
        switch (tensor->getComponentType().getKind()) {
          case ComponentType::Bool:
            curVal.bval = tensor->getStorage().getValues()[idx];
            break;
          case ComponentType::Int:
            curVal.ival = tensor->getStorage().getValues()[idx];
            break;
          case ComponentType::Float:
            curVal.fval = tensor->getStorage().getValues()[idx];
            break;
          case ComponentType::Double:
            curVal.dval = tensor->getStorage().getValues()[idx];
            break;
          default:
            taco_not_supported_yet;
            break;
        }

        for (size_t i = 0; i < lvl; ++i) {
          const size_t dim = levels[i].getDimension();
          curVal.loc[dim] = coord[i];
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

    const TensorBase* tensor;
    Coordinate::Coord coord;
    Coordinate::Coord ptrs;
    Coordinate        curVal;
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
  struct Content;

  void compute(bool pack);

  friend std::ostream& operator<<(std::ostream&, const TensorBase&);

  std::shared_ptr<Content> content;
};

}
#endif
