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

    friend std::ostream& operator<<(std::ostream& o, const AxisMatch& m) {
      switch (m.kind) {
        case Axis: { return o << "Axis(" << m.axis << ")"; }
        case Face: { return o << "Face(" << m.face << ")"; }
        case Replicated: { return o << "Replicate()"; }
        default: taco_iassert(false);
      }
      return o;
    }
  };

  GridPlacement(std::vector<AxisMatch> axes) : axes(axes) {}

  friend std::ostream& operator<<(std::ostream& o, const GridPlacement& gp) {
    return o << util::join(gp.axes);
  }

  std::vector<AxisMatch> axes;
};

GridPlacement::AxisMatch Face(int face);
GridPlacement::AxisMatch Replicate();

struct PlacementGrid {
  struct Axis {
    ir::Expr dimension;

    bool axisSet = false;
    GridPlacement::AxisMatch axis;

    Axis(ir::Expr dimension) : dimension(dimension) {}
    Axis(ir::Expr dimension, GridPlacement::AxisMatch axis) : dimension(dimension), axisSet(true), axis(axis) {}
  };

  template <typename... Args>
  PlacementGrid(Args... args) : PlacementGrid(std::vector<Axis>{args...}) {}

  PlacementGrid(std::vector<Axis> axes) {
    std::vector<ir::Expr> dims;
    std::vector<GridPlacement::AxisMatch> axisMatches;
    // If all of the axes values are unset or not Axis(n), then we can assume
    // the default case.
    bool unset = true;
    for (auto& axis : axes) {
      if (axis.axisSet && axis.axis.kind == GridPlacement::AxisMatch::Axis) {
        unset = false;
      }
    }
    // If all of the axes are unset, manually add the implied axes. Otherwise,
    // assert that all of the axes are set.
    if (unset) {
      int counter = 0;
      for (size_t i = 0; i < axes.size(); i++) {
        if (!axes[i].axisSet) {
          axes[i].axis = GridPlacement::AxisMatch(counter);
          axes[i].axisSet = true;
          counter++;
        }
      }
    } else {
      for (auto& axis : axes) {
        taco_uassert(axis.axisSet);
      }
    }

    for (auto& axis : axes) {
      dims.push_back(axis.dimension);
      axisMatches.push_back(axis.axis);
    }

    this->grid = Grid(dims);
    this->placement = GridPlacement(axisMatches);
  }

  friend std::ostream& operator<<(std::ostream& o, const PlacementGrid& pg) {
    return o << pg.grid << " <-> " << pg.placement << std::endl;
  }

  Grid getPartitionGrid() {
    std::vector<ir::Expr> dims;
    for (auto& axis : this->placement.axes) {
      if (axis.kind == GridPlacement::AxisMatch::Axis) {
        dims.push_back(this->grid.getDimSize(axis.axis));
      }
    }
    return Grid(dims);
  }

  Grid grid;
  GridPlacement placement;
};

PlacementGrid::Axis operator|(ir::Expr e, GridPlacement::AxisMatch axis);

struct TensorDistributionV2 {
  PlacementGrid pg;
  ParallelUnit parUnit;

  TensorDistributionV2(PlacementGrid pg, ParallelUnit parUnit = ParallelUnit::DistributedNode)
    : pg(pg), parUnit(parUnit) {}
};

// Struct that represents a level of distribution for a tensor.
struct TensorDistribution {
  Grid partitionGrid;
  Grid placementGrid;
  GridPlacement placement;
  ParallelUnit parUnit;

  // Simple use case constructor that partitions and places a tensor onto an
  // n-dimensional grid.
  explicit TensorDistribution(Grid g, ParallelUnit pu = ParallelUnit::DistributedNode) :
      partitionGrid(g), placementGrid(g), parUnit(pu) {
    // Construct a default placement object.
    std::vector<GridPlacement::AxisMatch> placements(g.getDim());
    for (int i = 0; i < g.getDim(); i++) {
      placements[i] = i;
    }
    this->placement = GridPlacement(placements);
  }

  // Full constructor that defines all needed fields.
  TensorDistribution(Grid partitionGrid, Grid placementGrid, GridPlacement placement, ParallelUnit pu = ParallelUnit::DistributedNode) :
      partitionGrid(partitionGrid), placementGrid(placementGrid), placement(placement), parUnit(pu) {}

  TensorDistribution(TensorDistributionV2 v2) :
    partitionGrid(v2.pg.getPartitionGrid()), placementGrid(v2.pg.grid), placement(v2.pg.placement), parUnit(v2.parUnit) {}
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
