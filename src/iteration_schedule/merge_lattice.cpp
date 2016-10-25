#include "merge_lattice.h"

#include "merge_rule.h"
#include "tensor_path.h"
#include "util/collections.h"
#include "util/strings.h"

using namespace std;

namespace taco {
namespace is {

// class MergeLatticePoint
MergeLatticePoint::MergeLatticePoint(std::vector<TensorPath> paths)
    : paths(paths) {
}

MergeLatticePoint::MergeLatticePoint(const TensorPath& path)
    : MergeLatticePoint(vector<TensorPath>({path})) {
}

const std::vector<TensorPath>& MergeLatticePoint::getPaths() const {
  return paths;
}

MergeLatticePoint operator+(MergeLatticePoint a, MergeLatticePoint b) {
  vector<TensorPath> paths;
  paths.insert(paths.end(), a.getPaths().begin(), a.getPaths().end());
  paths.insert(paths.end(), b.getPaths().begin(), b.getPaths().end());
  return MergeLatticePoint(paths);
}

std::ostream& operator<<(std::ostream& os, const MergeLatticePoint& mlp) {
  if (mlp.getPaths().size() > 1) {
    os << "(";
  }
  os << util::join(mlp.getPaths(), " \u2227 ");
  if (mlp.getPaths().size() > 1) {
    os << ")";
  }
  return os;
}


// class MergeLattice
MergeLattice::MergeLattice() {
}

MergeLattice::MergeLattice(std::vector<MergeLatticePoint> points)
    : points(points) {
}

MergeLattice::MergeLattice(MergeLatticePoint point)
    : MergeLattice(vector<MergeLatticePoint>({point})) {
}

const std::vector<MergeLatticePoint>& MergeLattice::getPoints() const {
  return points;
}

MergeLattice operator+(MergeLattice a, MergeLattice b) {
  vector<MergeLatticePoint> points;
  auto& aLatticePoints = a.getPoints();
  auto& bLatticePoints = b.getPoints();

  // Add all combinations of a and b lattice points
  for (auto& aLatticePoint : aLatticePoints) {
    for (auto& bLatticePoint : bLatticePoints) {
      points.push_back(aLatticePoint + bLatticePoint);
    }
  }

  // Append a lattice points
  util::append(points, aLatticePoints);

  // Append b lattice points
  util::append(points, bLatticePoints);


//  points.insert(points.end(), a.getPoints().begin(), a.getPoints().end());
//  points.insert(points.end(), b.getPoints().begin(), b.getPoints().end());
  return MergeLattice(points);
}

MergeLattice operator*(MergeLattice a, MergeLattice b) {
}

std::ostream& operator<<(std::ostream& os, const MergeLattice& ml) {
  return os << util::join(ml.getPoints(), " \u2228 ");
}


// functions
MergeLattice buildMergeLattice(const MergeRule& rule) {
  struct MergeLatticeVisitor : public MergeRuleVisitor {

    MergeLattice mergeLattice;
    MergeLattice buildMergeLattice(const MergeRule& rule) {
      rule.accept(this);
      return mergeLattice;
    }

    void visit(const Path* rule) {
      mergeLattice = MergeLatticePoint(rule->path);
    }

    void visit(const And* rule) {
      MergeLattice a = buildMergeLattice(rule->a);
      MergeLattice b = buildMergeLattice(rule->b);
    }

    void visit(const Or* rule) {
      MergeLattice a = buildMergeLattice(rule->a);
      MergeLattice b = buildMergeLattice(rule->b);
      mergeLattice = a + b;

//      LatticePoint
//      latticePoints = a;
//      latticePoints.insert(latticePoints.end(), b.begin(), b.end());
    }
  };
  MergeLattice mergeLattice = MergeLatticeVisitor().buildMergeLattice(rule);

  std::cout << std::endl << "# Lattice" << std::endl;
  std::cout << mergeLattice << std::endl;

  return mergeLattice;
}

}}
