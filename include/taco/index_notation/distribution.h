#ifndef TACO_DISTRIBUTION_H
#define TACO_DISTRIBUTION_H

#include <vector>
#include "taco/index_notation/index_notation.h"
#include "taco/util/strings.h"
#include "taco/ir/ir.h"

namespace taco {

// Grid represents an n-dimensional grid.
class Grid {
public:
  Grid() {};

  // Variadic template style constructor for grids.
  template <typename... Args>
  Grid(Args... args) : Grid(std::vector<ir::Expr>{args...}) {}

  Grid(std::vector<ir::Expr>& dims) {
    this->dimensions = dims;
  }

  Grid(std::vector<ir::Expr>&& dims) {
    this->dimensions = dims;
  }

  int getDim() {
    return this->dimensions.size();
  }

  ir::Expr getDimSize(int dim) {
    return this->dimensions[dim];
  }

  bool defined() { return this->dimensions.size() > 0; }

  friend std::ostream& operator<<(std::ostream& o, const Grid& g) {
    o << "Grid(" << util::join(g.dimensions) << ")";
    return o;
  }

private:
  std::vector<ir::Expr> dimensions;
};

class GridPlacement {
public:
  GridPlacement() = default;

  struct AxisMatch {
    enum Kind {
      Axis,
      Face,
      Replicated,
    };
    Kind kind;

    // Used when kind == Axis.
    int axis;

    // Used when kind == Face.
    int face;

    AxisMatch() = default;

    AxisMatch(int axis) : kind(Axis), axis(axis) {};

    static AxisMatch makeFace(int face) {
      AxisMatch a;
      a.kind = Face;
      a.face = face;
      return a;
    }

    static AxisMatch makeReplicated() {
      AxisMatch a;
      a.kind = Replicated;
      return a;
    }
  };

  GridPlacement(std::vector<AxisMatch> axes) : axes(axes) {}

  std::vector<AxisMatch> axes;
};

GridPlacement::AxisMatch Face(int face);
GridPlacement::AxisMatch Replicate();

// Struct that represents a level of distribution for a tensor.
struct TensorDistribution {
  Grid partitionGrid;
  Grid placementGrid;
  GridPlacement placement;
  ParallelUnit parUnit;
};

// Transfer represents requesting a portion of data.
// TODO (rohany): It seems like we're doing all equality on tensorvars, rather than the access.
//  That is fine, will just need to remember.
class Transfer {
public:
  Transfer() : content(nullptr) {};
  Transfer(taco::Access a);
  Access getAccess() const;
  friend bool operator==(Transfer& a, Transfer&b);
private:
  struct Content;
  std::shared_ptr<Content> content;
};
typedef std::vector<Transfer> Transfers;
std::ostream& operator<<(std::ostream&, const Transfer&);

}

#endif //TACO_DISTRIBUTION_H
