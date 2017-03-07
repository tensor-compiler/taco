#ifndef TACO_INTERNAL_TENSOR_H
#define TACO_INTERNAL_TENSOR_H

#include <memory>
#include <string>
#include <vector>

#include <iostream>

#include "format.h"
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
  TensorBase();
  TensorBase(std::string name, ComponentType ctype);
  TensorBase(std::string name, std::vector<int> dimensions,
             Format format, ComponentType, size_t);

  std::string getName() const;
  size_t getOrder() const;
  const std::vector<int>& getDimensions() const;
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

  void pack();
  void compile();
  void assemble();
  void compute();

  void setExpr(taco::Expr expr);
  void setIndexVars(std::vector<taco::Var> indexVars);

  void printIterationSpace() const;
  void printIR(std::ostream&) const;

  void printComputeIR(std::ostream&) const;
  void printAssemblyIR(std::ostream&) const;

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
