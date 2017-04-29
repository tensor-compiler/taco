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
  /// Create a scalar double
  TensorBase();

  /// Create a scalar
  TensorBase(ComponentType ctype);

  /// Create a scalar with the given name
  TensorBase(std::string name, ComponentType ctype);

  /// Create a tensor. The format defaults to sparse in every dimension, but
  /// can be changed with the `setFormat` method prior to packing.
  TensorBase(std::string name, ComponentType ctype,
             std::vector<int> dimensions);

  /// Create a tensor. The format defaults to sparse in every dimension, but
  /// can be changed with the `setFormat` method prior to packing.
  TensorBase(ComponentType ctype, std::vector<int> dimensions);

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
  const ComponentType& getComponentType() const;

  /// Get the format the tensor is packed into
  const Format& getFormat() const;

  const std::vector<taco::Var>& getIndexVars() const;
  const taco::Expr& getExpr() const;

  const storage::Storage& getStorage() const;
  storage::Storage getStorage();

  size_t getAllocSize() const;

  /// Set a new tensor format
  void setFormat(Format format);

  /// Reserve space for `numCoordinates` additional coordinates.
  void reserve(size_t numCoordinates) {
    size_t newSize =
        this->coordinateBuffer->size() + numCoordinates*this->coordinateSize;
    this->coordinateBuffer->resize(newSize);
  }

  /// Insert a value into the tensor. The number of coordinates must match the
  /// tensor dimension.
  void insert(const std::initializer_list<int>& coordinate, double value) {
    taco_uassert(coordinate.size() == getOrder()) << "Wrong number of indices";
    taco_uassert(getComponentType() == ComponentType::Double) <<
        "Cannot insert a value of type '" << ComponentType::Double << "' " <<
        "into a tensor with component type " << getComponentType();
    if ((coordinateBuffer->size() - coordinateBufferUsed) < coordinateSize) {
      coordinateBuffer->resize(coordinateBuffer->size() + coordinateSize);
    }
    int* coordLoc = (int*)&coordinateBuffer->data()[coordinateBufferUsed];
    for (int idx : coordinate) {
      *coordLoc = idx;
      coordLoc++;
    }
    *((double*)coordLoc) = value;
    coordinateBufferUsed += coordinateSize;
  }

  /// Insert a value into the tensor. The number of coordinates must match the
  /// tensor dimension.
  void insert(const std::vector<int>& coordinate, double value) {
    taco_uassert(coordinate.size() == getOrder()) << "Wrong number of indices";
    taco_uassert(getComponentType() == ComponentType::Double) <<
        "Cannot insert a value of type '" << ComponentType::Double << "' " <<
        "into a tensor with component type " << getComponentType();
    if ((coordinateBuffer->size() - coordinateBufferUsed) < coordinateSize) {
      coordinateBuffer->resize(coordinateBuffer->size() + coordinateSize);
    }
    int* coordLoc = (int*)&coordinateBuffer->data()[coordinateBufferUsed];
    for (int idx : coordinate) {
      *coordLoc = idx;
      coordLoc++;
    }
    *((double*)coordLoc) = value;
    coordinateBufferUsed += coordinateSize;
  }

  void setCSR(double* vals, int* rowPtr, int* colIdx);
  void getCSR(double** vals, int** rowPtr, int** colIdx);

  void setCSC(double* vals, int* colPtr, int* rowIdx);
  void getCSC(double** vals, int** colPtr, int** rowIdx);

  /// Pack tensor into the given format
  void pack();

  /// Zero out the values
  void zero();

  /// Compile the tensor expression.
  void compile();

  /// Assemble the tensor storage, including index and value arrays.
  void assemble();

  /// Compute the given expression and put the values in the tensor storage.
  void compute();

  /// Compile, assemble and compute as needed.
  void evaluate();

  void setExpr(taco::Expr expr);
  void setIndexVars(std::vector<taco::Var> indexVars);

  void printComputeIR(std::ostream&, bool color=false) const;
  void printAssemblyIR(std::ostream&, bool color=false) const;

  /// Get the source code of the kernel functions.
  std::string getSource() const;

  /// Compile the source code of the kernel functions. This function is optional
  /// and mainly intended for experimentation. If the source code is not set
  /// then it will will be created it from the given expression.
  void compileSource(std::string source);

  friend bool operator==(const TensorBase&, const TensorBase&);
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

  // True iff two tensors have the same type and the same values.
  friend bool equals(const TensorBase&, const TensorBase&);

  friend std::ostream& operator<<(std::ostream&, const TensorBase&);

private:
  struct Content;
  std::shared_ptr<Content> content;

  std::shared_ptr<std::vector<char>> coordinateBuffer;
  size_t                             coordinateBufferUsed;
  size_t                             coordinateSize;

  void assembleInternal();
  void computeInternal();
};

/// The file formats supported by the taco file readers and writers.
enum class FileFormat {
  /// .dns - A dense tensor format. It consists of zero or more lines of
  ///        comments preceded by '%'. Values are stored row major and separated
  ///        by whitespace.
  dns,

  /// .tns - The frostt sparse tensor format.  It consists of zero or more
  ///        comment lines preceded by '#', followed by any number of lines with
  ///        one coordinate/value per line.  The tensor dimensions are inferred
  ///        from the largest coordinates.
  tns,

  /// .mtx - The matrix market sparse matrix format.  It consists of a header
  ///        line preceded by '%%', zero or more comment lines preceded by '%',
  ///        a line with the number of rows, the number of columns and the
  //         number of non-zeroes, and any number of lines with one
  ///        coordinate/value per line.
  mtx,

  /// .rb  - The rutherford-boeing sparse matrix format.
  rb
};

/// Read a tensor from a file. The file format is inferred from the filename.
TensorBase readTensor(std::string filename, std::string name="");

/// Read a tensor from a file of the given file format.
TensorBase readTensor(std::string filename, FileFormat fileFormat,
                std::string name="");

/// Read a tensor from a stream of the given file format.
TensorBase readTensor(std::istream& stream, FileFormat fileFormat,
                std::string name="");

/// Write a tensor to a file. The file format is inferred from the filename.
void writeTensor(std::string filename, const TensorBase& tensor);

/// Write a tensor to a file in the given file format.
void writeTensor(std::string filename, FileFormat format,
                 const TensorBase& tensor);

/// Write a tensor to a stream in the given file format.
void writeTensor(std::ofstream& file, FileFormat format,
                 const TensorBase& tensor);

/// Pack the operands in the given expression.
void packOperands(const TensorBase& tensor);

}
#endif
