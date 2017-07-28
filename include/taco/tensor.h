#ifndef TACO_TENSOR_H
#define TACO_TENSOR_H

#include <memory>
#include <string>
#include <vector>
#include <cassert>

#include "taco/type.h"
#include "taco/expr.h"
#include "taco/format.h"
#include "taco/error.h"

#include "taco/storage/storage.h"
#include "taco/storage/index.h"
#include "taco/storage/array.h"
#include "taco/storage/array_util.h"

namespace taco {

/// TensorBase is the super-class for all tensors. You can use it directly to
/// avoid templates, or you can use the templated `Tensor<T>` that inherits from
/// `TensorBase`.
class TensorBase {
public:
  /// Create a scalar double
  TensorBase();

  /// Create a scalar
  TensorBase(Type ctype);

  /// Create a scalar with the given name
  TensorBase(std::string name, Type ctype);

  /// Create a scalar double
  explicit TensorBase(double);

  /// Create a tensor with the given dimensions and format. The format defaults
  // to sparse in every mode.
  TensorBase(Type ctype, std::vector<int> dimensions, Format format=Sparse);

  /// Create a tensor with the given dimensions and format. The format defaults
  // to sparse in every mode.
  TensorBase(std::string name, Type ctype, std::vector<int> dimensions,
             Format format=Sparse);

  /// Set the name of the tensor.
  void setName(std::string name) const;

  /// Get the name of the tensor.
  std::string getName() const;

  /// Get the order of the tensor (the number of modes).
  size_t getOrder() const;

  /// Get the size of a tensor mode.
  int getDimension(size_t dim) const;

  /// Get a vector with the size of each tensor mode.
  const std::vector<int>& getDimensions() const;

  /// Return the type of the tensor components (e.g. double).
  const Type& getComponentType() const;

  /// Get the format the tensor is packed into
  const Format& getFormat() const;

  /// Reserve space for `numCoordinates` additional coordinates.
  void reserve(size_t numCoordinates);

  /// Insert a value into the tensor. The number of coordinates must match the
  /// tensor order.
  void insert(const std::initializer_list<int>& coordinate, double value);

  /// Insert a value into the tensor. The number of coordinates must match the
  /// tensor order.
  void insert(const std::vector<int>& coordinate, double value);

  /// Returns the storage for this tensor. Tensor values are stored according
  /// to the format of the tensor.
  const storage::Storage& getStorage() const;

  /// Returns the storage for this tensor. Tensor values are stored according
  /// to the format of the tensor.
  storage::Storage& getStorage();

  /// Pack tensor into the given format
  void pack();

  /// Zero out the values
  void zero();

  const std::vector<taco::IndexVar>& getIndexVars() const;
  const taco::IndexExpr& getExpr() const;

  /// Create an index expression that accesses (reads) this tensor.
  const Access operator()(const std::vector<IndexVar>& indices) const;

  /// Create an index expression that accesses (reads or writes) this tensor.
  Access operator()(const std::vector<IndexVar>& indices);

  /// Create an index expression that accesses (reads) this tensor.
  template <typename... IndexVars>
  const Access operator()(const IndexVars&... indices) const {
    return static_cast<const TensorBase*>(this)->operator()({indices...});
  }

  /// Create an index expression that accesses (reads or writes) this tensor.
  template <typename... IndexVars>
  Access operator()(const IndexVars&... indices) {
    return this->operator()({indices...});
  }

  /// Set the expression to be evaluated when calling compute or assemble.
  void setExpr(const std::vector<taco::IndexVar>& indexVars,
               taco::IndexExpr expr);

  /// Compile the tensor expression.
  void compile(bool assembleWhileCompute=false);

  /// Assemble the tensor storage, including index and value arrays.
  void assemble();

  /// Compute the given expression and put the values in the tensor storage.
  void compute();

  /// Compile, assemble and compute as needed.
  void evaluate();

  /// Get the source code of the kernel functions.
  std::string getSource() const;

  /// Compile the source code of the kernel functions. This function is optional
  /// and mainly intended for experimentation. If the source code is not set
  /// then it will will be created it from the given expression.
  void compileSource(std::string source);

  /// Print the IR loops that compute the tensor's expression.
  void printComputeIR(std::ostream& stream, bool color=false,
                      bool simplify=false) const;

  /// Print the IR loops that assemble the tensor's expression.
  void printAssembleIR(std::ostream& stream, bool color=false,
                       bool simplify=false) const;

  /// Set the size of the initial index allocations.  The default size is 1MB.
  void setAllocSize(size_t allocSize);

  /// Get the size of the initial index allocations.
  size_t getAllocSize() const;

  /// True iff two tensors have the same type and the same values.
  friend bool equals(const TensorBase&, const TensorBase&);

  /// True iff two TensorBase objects refer to the same tensor (TensorBase
  /// and Tensor objects are references to tensors).
  friend bool operator==(const TensorBase& a, const TensorBase& b);
  friend bool operator!=(const TensorBase& a, const TensorBase& b);

  /// True iff the address of the tensor referenced by a is smaller than the
  /// address of b.  This is arbitrary and non-deterministic, but necessary for
  /// tensor to be placed in maps.
  friend bool operator<(const TensorBase& a, const TensorBase& b);
  friend bool operator>(const TensorBase& a, const TensorBase& b);
  friend bool operator<=(const TensorBase& a, const TensorBase& b);
  friend bool operator>=(const TensorBase& a, const TensorBase& b);

  /// Print a tensor to a stream.
  friend std::ostream& operator<<(std::ostream&, const TensorBase&);

private:
  struct Content;
  std::shared_ptr<Content> content;

  std::shared_ptr<std::vector<char>> coordinateBuffer;
  size_t                             coordinateBufferUsed;
  size_t                             coordinateSize;
};


/// A reference to a tensor. Tensor object copies copies the reference, and
/// subsequent method calls affect both tensor references. To deeply copy a
/// tensor (for instance to change the format) compute a copy index expression
/// e.g. `A(i,j) = B(i,j).
template <typename CType>
class Tensor : public TensorBase {
public:
  /// Create a scalar
  Tensor() : TensorBase() {}

  /// Create a scalar with the given name
  explicit Tensor(std::string name) : TensorBase(name, type<CType>()) {}

  /// Create a scalar double
  explicit Tensor(CType value) : TensorBase(value) {}

  /// Create a tensor with the given dimensions and format
  Tensor(std::vector<int> dimensions, Format format=Sparse)
      : TensorBase(type<CType>(), dimensions, format) {}

  /// Create a tensor with the given name, dimensions and format
  Tensor(std::string name, std::vector<int> dimensions, Format format)
      : TensorBase(name, type<CType>(), dimensions, format) {}

  /// Create a tensor from a TensorBase instance. The Tensor and TensorBase
  /// objects will reference the same underlying tensor so it is a shallow copy.
  Tensor(const TensorBase& tensor) : TensorBase(tensor) {
    taco_uassert(tensor.getComponentType() == type<CType>()) <<
        "Assigning TensorBase with " << tensor.getComponentType() <<
        " components to a Tensor<" << type<CType>() << ">";
  }

  class const_iterator {
  public:
    typedef const_iterator self_type;
    typedef std::pair<std::vector<int>,CType>  value_type;
    typedef std::pair<std::vector<int>,CType>& reference;
    typedef std::pair<std::vector<int>,CType>* pointer;
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

    const std::pair<std::vector<int>,CType>& operator*() const {
      return curVal;
    }

    const std::pair<std::vector<int>,CType>* operator->() const {
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
        coord(std::vector<int>(tensor->getOrder())),
        ptrs(std::vector<int>(tensor->getOrder())),
        curVal({std::vector<int>(tensor->getOrder()), 0}),
        count(1 + (size_t)isEnd * tensor->getStorage().getIndex().getSize()),
        advance(false) {
      advanceIndex();
    }

    void advanceIndex() {
      advanceIndex(0);
      ++count;
    }

    bool advanceIndex(size_t lvl) {
      using namespace taco::storage;

      const auto& modeTypes = tensor->getFormat().getModeTypes();
      const auto& modeOrder = tensor->getFormat().getModeOrder();

      if (lvl == tensor->getOrder()) {
        if (advance) {
          advance = false;
          return false;
        }

        const size_t idx = (lvl == 0) ? 0 : ptrs[lvl - 1];
        curVal.second = getValue<double>(tensor->getStorage().getValues(), idx);

        for (size_t i = 0; i < lvl; ++i) {
          const size_t dim = modeOrder[i];
          curVal.first[dim] = coord[i];
        }

        advance = true;
        return true;
      }
      
      const auto storage = tensor->getStorage();
      const auto modeIndex = storage.getIndex().getModeIndex(lvl);

      switch (modeTypes[lvl]) {
        case Dense: {
          const auto size = getValue<int>(modeIndex.getIndexArray(0), 0);
          const auto base = (lvl == 0) ? 0 : (ptrs[lvl - 1] * size);

          if (advance) {
            goto resume_dense;  // obligatory xkcd: https://xkcd.com/292/
          }

          for (coord[lvl] = 0; coord[lvl] < size; ++coord[lvl]) {
            ptrs[lvl] = base + coord[lvl];

          resume_dense:
            if (advanceIndex(lvl + 1)) {
              return true;
            }
          }
          break;
        }
        case Sparse: {
          const auto& pos = modeIndex.getIndexArray(0);
          const auto& idx = modeIndex.getIndexArray(1);
          const auto  k   = (lvl == 0) ? 0 : ptrs[lvl - 1];

          if (advance) {
            goto resume_sparse;
          }

          for (ptrs[lvl] = getValue<int>(pos, k);
               ptrs[lvl] < getValue<int>(pos, k+1);
               ++ptrs[lvl]) {
            coord[lvl] = getValue<int>(idx, ptrs[lvl]);

          resume_sparse:
            if (advanceIndex(lvl + 1)) {
              return true;
            }
          }
          break;
        }
        case Fixed: {
          const auto  elems = getValue<int>(modeIndex.getIndexArray(0), 0);
          const auto  base  = (lvl == 0) ? 0 : (ptrs[lvl - 1] * elems);
          const auto& vals  = modeIndex.getIndexArray(1);

          if (advance) {
            goto resume_fixed;
          }

          for (ptrs[lvl] = base;
               ptrs[lvl] < base + elems && getValue<int>(vals, ptrs[lvl]) >= 0;
               ++ptrs[lvl]) {
            coord[lvl] = getValue<int>(vals, ptrs[lvl]);

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

    const Tensor<CType>*              tensor;
    std::vector<int>                  coord;
    std::vector<int>                  ptrs;
    std::pair<std::vector<int>,CType> curVal;
    size_t                            count;
    bool                              advance;
  };

  const_iterator begin() const {
    return const_iterator(this);
  }

  const_iterator end() const {
    return const_iterator(this, true);
  }
};


/// The file formats supported by the taco file readers and writers.
enum class FileType {
  /// .tns - The frostt sparse tensor format.  It consists of zero or more
  ///        comment lines preceded by '#', followed by any number of lines with
  ///        one coordinate/value per line.  The tensor dimensions are inferred
  ///        from the largest coordinates.
  tns,

  /// .mtx - The matrix market matrix format.  It consists of a header
  ///        line preceded by '%%', zero or more comment lines preceded by '%',
  ///        a line with the number of rows, the number of columns and the
  //         number of non-zeroes. For sparse matrix and any number of lines
  ///        with one coordinate/value per line, and for dense a list of values.
  mtx,

  /// .ttx - The tensor format derived from matrix market format. It consists
  ///        with the same header file and coordinates/values list.
  ttx,

  /// .rb  - The rutherford-boeing sparse matrix format.
  rb
};

/// Read a tensor from a file. The file format is inferred from the filename
/// and the tensor is returned packed by default.
TensorBase read(std::string filename, Format format, bool pack = true);

/// Read a tensor from a file of the given file format and the tensor is
/// returned packed by default.
TensorBase read(std::string filename, FileType filetype, Format format,
                bool pack = true);

/// Read a tensor from a stream of the given file format. The tensor is returned
/// packed by default.
TensorBase read(std::istream& stream, FileType filetype, Format format,
                bool pack = true);

/// Write a tensor to a file. The file format is inferred from the filename.
void write(std::string filename, const TensorBase& tensor);

/// Write a tensor to a file in the given file format.
void write(std::string filename, FileType filetype, const TensorBase& tensor);

/// Write a tensor to a stream in the given file format.
void write(std::ofstream& file, FileType filetype, const TensorBase& tensor);


/// Factory function to construct a compressed sparse row (CSR) matrix. The
/// arrays remain owned by the user and will not be freed by taco.
TensorBase makeCSR(const std::string& name, const std::vector<int>& dimensions,
                   int* rowptr, int* colidx, double* vals);

/// Factory function to construct a compressed sparse row (CSR) matrix.
TensorBase makeCSR(const std::string& name, const std::vector<int>& dimensions,
                   const std::vector<int>& rowptr,
                   const std::vector<int>& colidx,
                   const std::vector<double>& vals);

/// Get the arrays that makes up a compressed sparse row (CSR) tensor. This
/// function does not change the ownership of the arrays.
void getCSRArrays(const TensorBase& tensor,
                  int** rowptr, int** colidx, double** vals);

/// Factory function to construct a compressed sparse columns (CSC) matrix. The
/// arrays remain owned by the user and will not be freed by taco.
TensorBase makeCSC(const std::string& name, const std::vector<int>& dimensions,
                   int* colptr, int* rowidx, double* vals);

/// Factory function to construct a compressed sparse columns (CSC) matrix.
TensorBase makeCSC(const std::string& name, const std::vector<int>& dimensions,
                   const std::vector<int>& colptr,
                   const std::vector<int>& rowidx,
                   const std::vector<double>& vals);

/// Get the arrays that makes up a compressed sparse columns (CSC) tensor. This
/// function does not change the ownership of the arrays.
void getCSCArrays(const TensorBase& tensor,
                  int** colptr, int** rowidx, double** vals);


/// Pack the operands in the given expression.
void packOperands(const TensorBase& tensor);

/// Iterate over the typed values of a TensorBase.
template <typename CType>
Tensor<CType> iterate(const TensorBase& tensor) {
  return Tensor<CType>(tensor);
}

}
#endif
