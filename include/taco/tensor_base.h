#ifndef TACO_INTERNAL_TENSOR_H
#define TACO_INTERNAL_TENSOR_H

#include <memory>
#include <string>
#include <vector>

#include <iostream>

#include "taco/format.h"
#include "component_types.h"
#include "util/comparable.h"
#include "util/strings.h"

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
  void readMTX(std::string filename);

  /// Pack tensor into the given format
  void pack();

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

  void printIterationSpace() const;
  void printIR(std::ostream&) const;

  void printComputeIR(std::ostream&, bool color) const;
  void printAssemblyIR(std::ostream&, bool color) const;

  friend bool operator!=(const TensorBase&, const TensorBase&);
  friend bool operator<(const TensorBase&, const TensorBase&);

private:
  struct Content;

  struct Coordinate : util::Comparable<Coordinate> {
    typedef std::vector<int> Coord;

    Coordinate(const Coord& loc, int    val) : loc(loc), ival(val) {}
    Coordinate(const Coord& loc, float  val) : loc(loc), fval(val) {}
    Coordinate(const Coord& loc, double val) : loc(loc), dval(val) {}
    Coordinate(const Coord& loc, bool   val) : loc(loc), bval(val) {}

    std::vector<int> loc;
    union {
      int    ival;
      float  fval;
      double dval;
      bool   bval;
    };

    friend bool operator==(const Coordinate& l, const Coordinate& r) {
      iassert(l.loc.size() == r.loc.size());
      return l.loc == r.loc;
    }
    friend bool operator<(const Coordinate& l, const Coordinate& r) {
      iassert(l.loc.size() == r.loc.size());
      return l.loc < r.loc;
    }
  };
  
  friend std::ostream& operator<<(std::ostream&, const TensorBase&);

  std::shared_ptr<Content> content;
};

}
#endif
