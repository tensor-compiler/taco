#ifndef TACO_TENSOR_BASE_H
#define TACO_TENSOR_BASE_H

#include <memory>
#include <string>
#include <vector>
#include <cassert>

#include "taco/expr.h"
#include "taco/format.h"
#include "taco/error.h"
#include "storage/storage.h"

namespace taco {

/// Tensor component types. These are basic types such as double and int.
class ComponentType {
public:
  enum Kind {Bool, Int, Float, Double, Unknown};
  ComponentType() : ComponentType(Unknown) {}
  ComponentType(Kind kind) : kind(kind)  {}
  size_t bytes() const;
  Kind getKind() const;
private:
  Kind kind;
};

bool operator==(const ComponentType& a, const ComponentType& b);
bool operator!=(const ComponentType& a, const ComponentType& b);
std::ostream& operator<<(std::ostream&, const ComponentType&);
template <typename T> inline ComponentType type() {
  assert(false && "Unsupported type");
  return ComponentType::Double;
}
template <> inline ComponentType type<bool>() {return ComponentType::Bool;}
template <> inline ComponentType type<int>() {return ComponentType::Int;}
template <> inline ComponentType type<float>() {return ComponentType::Float;}
template <> inline ComponentType type<double>() {return ComponentType::Double;}


/// TensorBase is the super-class for all tensors. It can be instantiated
/// directly, which avoids templates, or a templated  `Tensor<T>` that inherits
/// from `TensorBase` can be instantiated.
class TensorBase {
public:
  /// Create a scalar double
  TensorBase();

  /// Create a scalar
  TensorBase(ComponentType ctype);

  /// Create a scalar with the given name
  TensorBase(std::string name, ComponentType ctype);

  /// Create a tensor with the given dimensions and format. The format defaults
  // to sparse in every dimension
  TensorBase(ComponentType ctype, std::vector<int> dimensions,
             Format format=Sparse);

  /// Create a tensor with the given dimensions and format. The format defaults
  // to sparse in every dimension
  TensorBase(std::string name, ComponentType ctype, std::vector<int> dimensions,
             Format format=Sparse);

  /// Set the name of the tensor.
  void setName(std::string name) const;

  /// Get the name of the tensor.
  std::string getName() const;

  /// Get the order of the tensor (the number of dimensions/modes).
  size_t getOrder() const;

  /// Get a vector with the size of each tensor dimension/mode.
  const std::vector<int>& getDimensions() const;

  /// Return the type of the tensor components (e.g. double).
  const ComponentType& getComponentType() const;

  /// Get the format the tensor is packed into
  const Format& getFormat() const;

  /// Reserve space for `numCoordinates` additional coordinates.
  void reserve(size_t numCoordinates);

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

  /// Returns the storage for this tensor. Tensor values are stored according
  /// to the format of the tensor.
  const storage::Storage& getStorage() const;

  /// Returns the storage for this tensor. Tensor values are stored according
  /// to the format of the tensor.
  storage::Storage& getStorage();

  void setCSR(double* vals, int* rowPtr, int* colIdx);
  void getCSR(double** vals, int** rowPtr, int** colIdx);

  void setCSC(double* vals, int* colPtr, int* rowIdx);
  void getCSC(double** vals, int** colPtr, int** rowIdx);

  /// Pack tensor into the given format
  void pack();

  /// Zero out the values
  void zero();

  const std::vector<taco::Var>& getIndexVars() const;
  const taco::Expr& getExpr() const;

  /// Create an index expression that accesses (reads/writes) this tensor.
  Access operator()(const std::vector<Var>& indices);

  /// Create an index expression that accesses (reads/writes) this tensor.
  template <typename... Vars>
  Access operator()(const Vars&... indices) {
    return this->operator()({indices...});
  }

  /// Set the expression to be evaluated when calling compute or assemble.
  void setExpr(const std::vector<taco::Var>& indexVars, taco::Expr expr);

  /// Compile the tensor expression.
  void compile();

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
  void setAllocSize(size_t allocSize) const;

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

  void assembleInternal();
  void computeInternal();
};


/// The file formats supported by the taco file readers and writers.
enum class FileType {
  /// .dns - A dense tensor format. It consists of zero or more lines of
  ///        comments preceded by '#', followed by a header line with the size
  ///        of each dimension  followed by values that are stored row major and
  ///        separated by whitespace.
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


/// Pack the operands in the given expression.
void packOperands(const TensorBase& tensor);

}
#endif
