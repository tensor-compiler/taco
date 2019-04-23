#ifndef TACO_TENSOR_H
#define TACO_TENSOR_H

#include <memory>
#include <string>
#include <vector>
#include <cassert>
#include <utility>
#include <array>

#include "taco/type.h"
#include "taco/format.h"

#include "taco/codegen/module.h"

#include "taco/index_notation/index_notation.h"

#include "taco/storage/storage.h"
#include "taco/storage/index.h"
#include "taco/storage/array.h"
#include "taco/storage/typed_vector.h"
#include "taco/storage/typed_index.h"

#include "taco/util/name_generator.h"
#include "taco/error.h"
#include "taco/error/error_messages.h"


namespace taco {

/// TensorBase is the super-class for all tensors. You can use it directly to
/// avoid templates, or you can use the templated `Tensor<T>` that inherits from
/// `TensorBase`.
class TensorBase {
public:
  /// Create a scalar
  TensorBase();

  /// Create a scalar
  TensorBase(Datatype ctype);

  /// Create a scalar with the given name
  TensorBase(std::string name, Datatype ctype);

  /// Create a scalar
  template <typename T>
  explicit TensorBase(T val) : TensorBase(type<T>()) {
    this->insert({}, val);
    pack();
  }
  
  /// Create a tensor with the given dimensions. The format defaults to sparse 
  /// in every mode.
  TensorBase(Datatype ctype, std::vector<int> dimensions, 
             ModeFormat modeType = ModeFormat::compressed);
  
  /// Create a tensor with the given dimensions and format.
  TensorBase(Datatype ctype, std::vector<int> dimensions, Format format); 

  /// Create a tensor with the given data type, dimensions and format. The 
  /// format defaults to sparse in every mode.
  TensorBase(std::string name, Datatype ctype, std::vector<int> dimensions, 
             ModeFormat modeType = ModeFormat::compressed); 
  
  /// Create a tensor with the given data type, dimensions and format.
  TensorBase(std::string name, Datatype ctype, std::vector<int> dimensions,
             Format format);

  /// Set the name of the tensor.
  void setName(std::string name) const;

  /// Get the name of the tensor.
  std::string getName() const;

  /// Get the order of the tensor (the number of modes).
  int getOrder() const;

  /// Get the dimension of a tensor mode.
  int getDimension(int mode) const;

  /// Get a vector with the dimension of each tensor mode.
  const std::vector<int>& getDimensions() const;

  /// Return the type of the tensor components).
  const Datatype& getComponentType() const; 

  /// Get the format the tensor is packed into
  const Format& getFormat() const;

  /// Reserve space for `numCoordinates` additional coordinates.
  void reserve(size_t numCoordinates);

  /// Insert a value into the tensor. The number of coordinates must match the
  /// tensor order.
  template <typename T>
  void insert(const std::initializer_list<int>& coordinate, T value) {
    taco_uassert(coordinate.size() == (size_t)getOrder()) <<
    "Wrong number of indices";
    taco_uassert(getComponentType() == type<T>()) <<
    "Cannot insert a value of type '" << type<T>() << "' " <<
    "into a tensor with component type " << getComponentType();
    if ((coordinateBuffer->size() - coordinateBufferUsed) < coordinateSize) {
      coordinateBuffer->resize(coordinateBuffer->size() + coordinateSize);
    }
    int* coordLoc = (int*)&coordinateBuffer->data()[coordinateBufferUsed];
    for (int idx : coordinate) {
      *coordLoc = idx;
      coordLoc++;
    }
    TypedComponentPtr valLoc(getComponentType(), coordLoc);
    *valLoc = TypedComponentVal(getComponentType(), &value);
    coordinateBufferUsed += coordinateSize;
  }

  /// Insert a value into the tensor. The number of coordinates must match the
  /// tensor order.
  template <typename T>
  void insert(const std::vector<int>& coordinate, T value) {
    taco_uassert(coordinate.size() == (size_t)getOrder()) <<
    "Wrong number of indices";
    taco_uassert(getComponentType() == type<T>()) <<
      "Cannot insert a value of type '" << type<T>() << "' " <<
      "into a tensor with component type " << getComponentType();
    if ((coordinateBuffer->size() - coordinateBufferUsed) < coordinateSize) {
      coordinateBuffer->resize(coordinateBuffer->size() + coordinateSize);
    }
    int* coordLoc = (int*)&coordinateBuffer->data()[coordinateBufferUsed];
    for (int idx : coordinate) {
      *coordLoc = idx;
      coordLoc++;
    }
    TypedComponentPtr valLoc(getComponentType(), coordLoc);
    *valLoc = TypedComponentVal(getComponentType(), &value);
    coordinateBufferUsed += coordinateSize;
  }

  /// Pack tensor into the given format
  void pack();

  /// Set the tensor's storage
  void setStorage(TensorStorage storage);

  /// Returns the storage for this tensor. Tensor values are stored according
  /// to the format of the tensor.
  const TensorStorage& getStorage() const;

  /// Returns the storage for this tensor. Tensor values are stored according
  /// to the format of the tensor.
  TensorStorage& getStorage();

  /// Zero out the values
  void zero();

  /// Returns the tensor var for this tensor.
  const TensorVar& getTensorVar() const;

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

  /// Assign an expression to a scalar tensor.
  void operator=(const IndexExpr&);

  /// Set the expression to be evaluated when calling compute or assemble.
  void setAssignment(Assignment assignment);

  /// Set the expression to be evaluated when calling compute or assemble.
  Assignment getAssignment() const;

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

  /// Get the taco_tensor_t representation of this tensor.
  taco_tensor_t* getTacoTensorT();

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

protected:
  static std::shared_ptr<ir::Module> getHelperFunctions(
      const Format& format, Datatype ctype, const std::vector<int>& dimensions);

private:
  struct Content;
  std::shared_ptr<Content> content;

  std::shared_ptr<std::vector<char>> coordinateBuffer;
  size_t                             coordinateBufferUsed;
  size_t                             coordinateSize;

  static std::vector<std::tuple<Format,
                                Datatype,
                                std::vector<int>,
                                std::shared_ptr<ir::Module>>> helperFunctions;
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

  /// Create a scalar
  explicit Tensor(CType value) : TensorBase(value) {}

  /// Create a tensor with the given dimensions. The format defaults to sparse 
  /// in every mode.
  Tensor(std::vector<int> dimensions, ModeFormat modeType = ModeFormat::compressed) 
      : TensorBase(type<CType>(), dimensions) {}

  /// Create a tensor with the given dimensions and format
  Tensor(std::vector<int> dimensions, Format format)
      : TensorBase(type<CType>(), dimensions, format) {}

  /// Create a tensor with the given name, dimensions and format. The format 
  /// defaults to sparse in every mode.
  Tensor(std::string name, std::vector<int> dimensions, 
         ModeFormat modeType = ModeFormat::compressed)
      : TensorBase(name, type<CType>(), dimensions, modeType) {}

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

  /// Simple transpose that packs a new tensor from the values in the current tensor
  Tensor<CType> transpose(std::string name, std::vector<int> newModeOrdering) const {
    return transpose(name, newModeOrdering, getFormat());
  }
  Tensor<CType> transpose(std::vector<int> newModeOrdering) const {
    return transpose(util::uniqueName('A'), newModeOrdering);
  }
  Tensor<CType> transpose(std::vector<int> newModeOrdering, Format format) const {
    return transpose(util::uniqueName('A'), newModeOrdering, format);
  }
  Tensor<CType> transpose(std::string name, std::vector<int> newModeOrdering, Format format) const {
    // Reorder dimensions to match new mode ordering
    std::vector<int> newDimensions;
    for (int mode : newModeOrdering) {
      newDimensions.push_back(getDimensions()[mode]);
    }

    Tensor<CType> newTensor(name, newDimensions, format);
    for (const std::pair<std::vector<int>,CType>& value : *this) {
      std::vector<int> newCoordinate;
      for (int mode : newModeOrdering) {
        newCoordinate.push_back(value.first[mode]);
      }
      newTensor.insert(newCoordinate, value.second);
    }
    newTensor.pack();
    return newTensor;
  }

  template<typename T>
  class const_iterator {
  public:
    typedef const_iterator self_type;
    typedef std::pair<std::vector<T>,CType>  value_type;
    typedef std::pair<std::vector<T>,CType>& reference;
    typedef std::pair<std::vector<T>,CType>* pointer;
    typedef std::forward_iterator_tag iterator_category;

    const_iterator(const const_iterator& iterator) :
        tensor(iterator.tensor),
        tensorStorage(iterator.tensorStorage),
        tensorOrder(iterator.tensorOrder),
        bufferCapacity(iterator.bufferCapacity),
        bufferSize(iterator.bufferSize),
        bufferPos(iterator.bufferPos),
        iterFunc(iterator.iterFunc),
        ctx(iterator.ctx) {
    }

    const_iterator operator++() {
      advance();
      return *this;
    }

    const_iterator operator++(int) {
     const_iterator result = *this;
     advance();
     return result;
    }

    const value_type& operator*() const {
      return ctx->curVal;
    }

    const value_type* operator->() const {
      return &ctx->curVal;
    }

    bool operator==(const const_iterator& rhs) {
      return (tensor == rhs.tensor) && 
             (isEnd() == rhs.isEnd()) && 
             (isEnd() || ctx->curVal.first == rhs.ctx->curVal.first);
    }

    bool operator!=(const const_iterator& rhs) {
      return !(*this == rhs);
    }

  protected:
    bool isEnd() const {
      return (bufferSize == 0);
    }

  private:
    friend class Tensor;

    struct Context {
      Context(int order, int bufferCapacity, void* iterCtx) :
          coordBuffer(new T[order * bufferCapacity]),
          valBuffer(new CType[bufferCapacity]),
          iterCtx(iterCtx) {
        curVal.first.resize(order);
      }

      ~Context() {
        delete[] coordBuffer;
        delete[] valBuffer;
        free(iterCtx);
      }

      T* coordBuffer;
      CType* valBuffer;
      value_type curVal;
      void* iterCtx;
    };

    const_iterator(const Tensor<CType>* tensor, bool isEnd = false) :
        tensor(tensor),
        tensorStorage(tensor->getStorage()),
        tensorOrder(tensor->getOrder()),
        bufferCapacity(100),
        bufferSize(0),
        bufferPos(bufferSize),
        ctx(nullptr) {
      if (!isEnd) {
        ctx = std::make_shared<Context>(tensorOrder, bufferCapacity, nullptr);

        const auto helperFuncs = tensor->getHelperFunctions(tensor->getFormat(), 
            tensor->getComponentType(), tensor->getDimensions());
        *reinterpret_cast<void**>(&iterFunc) = 
            helperFuncs->getFuncPtr("_shim_iterate");

        advance();
      }
    }

    void advance() {
      if (ctx.use_count() > 1) {
        const int iterCtxSize = *((int*)ctx->iterCtx);
        auto ctxCopy = std::make_shared<Context>(tensorOrder, bufferCapacity, 
                                                 malloc(iterCtxSize));
        memcpy(ctxCopy->coordBuffer, ctx->coordBuffer, 
               tensorOrder * bufferCapacity * sizeof(T));
        memcpy(ctxCopy->valBuffer, ctx->valBuffer, 
               bufferCapacity * sizeof(CType));
        memcpy(ctxCopy->iterCtx, ctx->iterCtx, iterCtxSize);
        ctx = ctxCopy;
      }

      ++bufferPos;
      if (bufferPos >= bufferSize) {
        fillBuffer();
        bufferPos = 0;
      }

      memcpy(ctx->curVal.first.data(), 
             &(ctx->coordBuffer[bufferPos * tensorOrder]), 
             tensorOrder * sizeof(T));
      ctx->curVal.second = ctx->valBuffer[bufferPos];
    }

    void fillBuffer() {
      std::array<void*,5> args = {&ctx->iterCtx, ctx->coordBuffer, 
                                  ctx->valBuffer, (void*)&bufferCapacity, 
                                  (void*)tensorStorage};
      bufferSize = iterFunc(args.data());
    }

    typedef int (*fnptr_t)(void**);

    const Tensor<CType>*        tensor;
    const taco_tensor_t*        tensorStorage;
    const int                   tensorOrder;
    const int                   bufferCapacity;
    int                         bufferSize;
    int                         bufferPos;
    fnptr_t                     iterFunc;
    std::shared_ptr<Context>    ctx;
  };

  const_iterator<int> begin() const {
    return const_iterator<int>(this);
  }

  const_iterator<int> end() const {
    return const_iterator<int>(this, true);
  }

  template<typename T>
  const_iterator<T> beginTyped() const {
    return const_iterator<T>(this);
  }

  template<typename T>
  const_iterator<T> endTyped() const {
    return const_iterator<T>(this, true);
  }

  /// Assign an expression to a scalar tensor.
  void operator=(const IndexExpr& expr) {TensorBase::operator=(expr);}
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
TensorBase read(std::string filename, ModeFormat modeType, bool pack = true);

/// Read a tensor from a file. The file format is inferred from the filename
/// and the tensor is returned packed by default.
TensorBase read(std::string filename, Format format, bool pack = true);

/// Read a tensor from a file of the given file format and the tensor is
/// returned packed by default.
TensorBase read(std::string filename, FileType filetype, ModeFormat modetype,
                bool pack = true);

/// Read a tensor from a file of the given file format and the tensor is
/// returned packed by default.
TensorBase read(std::string filename, FileType filetype, Format format,
                bool pack = true);

/// Read a tensor from a stream of the given file format. The tensor is returned
/// packed by default.
TensorBase read(std::istream& stream, FileType filetype, ModeFormat modetype,
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

template<typename T>
TensorBase makeCSR(const std::string& name, const std::vector<int>& dimensions,
                   int* rowptr, int* colidx, T* vals) {
  taco_uassert(dimensions.size() == 2) << error::requires_matrix;
  Tensor<T> tensor(name, dimensions, CSR);
  auto storage = tensor.getStorage();
  auto index = makeCSRIndex(dimensions[0], rowptr, colidx);
  storage.setIndex(index);
  storage.setValues(makeArray(vals, index.getSize(), Array::UserOwns));
  return tensor;
}

/// Factory function to construct a compressed sparse row (CSR) matrix.
template<typename T>
TensorBase makeCSR(const std::string& name, const std::vector<int>& dimensions,
                   const std::vector<int>& rowptr,
                   const std::vector<int>& colidx,
                   const std::vector<T>& vals) {
  taco_uassert(dimensions.size() == 2) << error::requires_matrix;
  Tensor<T> tensor(name, dimensions, CSR);
  auto storage = tensor.getStorage();
  storage.setIndex(makeCSRIndex(rowptr, colidx));
  storage.setValues(makeArray(vals));
  return std::move(tensor);
}

/// Get the arrays that makes up a compressed sparse row (CSR) tensor. This
/// function does not change the ownership of the arrays.
template<typename T>
void getCSRArrays(const TensorBase& tensor,
                  int** rowptr, int** colidx, T** vals) {
  taco_uassert(tensor.getFormat() == CSR) <<
  "The tensor " << tensor.getName() << " is not defined in the CSR format";
  auto storage = tensor.getStorage();
  auto index = storage.getIndex();
  
  auto rowptrArr = index.getModeIndex(1).getIndexArray(0);
  auto colidxArr = index.getModeIndex(1).getIndexArray(1);
  taco_uassert(rowptrArr.getType() == type<int>()) << error::type_mismatch;
  taco_uassert(colidxArr.getType() == type<int>()) << error::type_mismatch;
  *rowptr = static_cast<int*>(rowptrArr.getData());
  *colidx = static_cast<int*>(colidxArr.getData());
  *vals   = static_cast<T*>(storage.getValues().getData());
}

/// Factory function to construct a compressed sparse columns (CSC) matrix. The
/// arrays remain owned by the user and will not be freed by taco.
template<typename T>
TensorBase makeCSC(const std::string& name, const std::vector<int>& dimensions,
                   int* colptr, int* rowidx, T* vals) {
  taco_uassert(dimensions.size() == 2) << error::requires_matrix;
  Tensor<T> tensor(name, dimensions, CSC);
  auto storage = tensor.getStorage();
  auto index = makeCSCIndex(dimensions[1], colptr, rowidx);
  storage.setIndex(index);
  storage.setValues(makeArray(vals, index.getSize(), Array::UserOwns));
  return tensor;
}

/// Factory function to construct a compressed sparse columns (CSC) matrix.
template<typename T>
TensorBase makeCSC(const std::string& name, const std::vector<int>& dimensions,
                   const std::vector<int>& colptr,
                   const std::vector<int>& rowidx,
                   const std::vector<T>& vals) {
  taco_uassert(dimensions.size() == 2) << error::requires_matrix;
  Tensor<T> tensor(name, dimensions, CSC);
  auto storage = tensor.getStorage();
  storage.setIndex(makeCSCIndex(colptr, rowidx));
  storage.setValues(makeArray(vals));
  return std::move(tensor);
}

/// Get the arrays that makes up a compressed sparse columns (CSC) tensor. This
/// function does not change the ownership of the arrays.
template<typename T>
void getCSCArrays(const TensorBase& tensor,
                  int** colptr, int** rowidx, T** vals) {
  taco_uassert(tensor.getFormat() == CSC) <<
  "The tensor " << tensor.getName() << " is not defined in the CSC format";
  auto storage = tensor.getStorage();
  auto index = storage.getIndex();
  
  auto colptrArr = index.getModeIndex(1).getIndexArray(0);
  auto rowidxArr = index.getModeIndex(1).getIndexArray(1);
  taco_uassert(colptrArr.getType() == type<int>()) << error::type_mismatch;
  taco_uassert(rowidxArr.getType() == type<int>()) << error::type_mismatch;
  *colptr = static_cast<int*>(colptrArr.getData());
  *rowidx = static_cast<int*>(rowidxArr.getData());
  *vals   = static_cast<T*>(storage.getValues().getData());
}


/// Pack the operands in the given expression.
void packOperands(const TensorBase& tensor);

/// Iterate over the typed values of a TensorBase.
template <typename CType>
Tensor<CType> iterate(const TensorBase& tensor) {
  return Tensor<CType>(tensor);
}

}
#endif
