#include "merge_lattice.h"

#include <algorithm>

#include "internal_tensor.h" //
#include "merge_rule.h"
#include "tensor_path.h"
#include "util/collections.h"
#include "util/strings.h"

using namespace std;

namespace taco {
namespace lower {

// class MergeLatticePoint
MergeLatticePoint::MergeLatticePoint(const TensorPathStep& step)
    : MergeLatticePoint(vector<TensorPathStep>({step})) {
}

MergeLatticePoint::MergeLatticePoint(std::vector<TensorPathStep> steps)
    : steps(steps) {
}

const std::vector<TensorPathStep>& MergeLatticePoint::getSteps() const {
  return steps;
}

MergeLatticePoint operator+(MergeLatticePoint a, MergeLatticePoint b) {
  vector<TensorPathStep> steps;
  steps.insert(steps.end(), a.getSteps().begin(), a.getSteps().end());
  steps.insert(steps.end(), b.getSteps().begin(), b.getSteps().end());
  return MergeLatticePoint(steps);
}

std::ostream& operator<<(std::ostream& os, const MergeLatticePoint& mlp) {
  vector<string> pathNames;
  if (mlp.getSteps().size() > 1) {
    os << "(";
  }
  os << util::join(mlp.getSteps(), " \u2227 ");
  if (mlp.getSteps().size() > 1) {
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

MergeLattice MergeLattice::make(const MergeRule& rule) {
  struct MergeLatticeVisitor : public MergeRuleVisitor {

    MergeLattice mergeLattice;
    MergeLattice buildMergeLattice(const MergeRule& rule) {
      rule.accept(this);
      return mergeLattice;
    }

    void visit(const Step* rule) {
      mergeLattice = MergeLatticePoint(rule->step);
    }

    void visit(const And* rule) {
      MergeLattice a = buildMergeLattice(rule->a);
      MergeLattice b = buildMergeLattice(rule->b);
      mergeLattice = a * b;
    }

    void visit(const Or* rule) {
      MergeLattice a = buildMergeLattice(rule->a);
      MergeLattice b = buildMergeLattice(rule->b);
      mergeLattice = a + b;
    }
  };
  MergeLattice mergeLattice = MergeLatticeVisitor().buildMergeLattice(rule);
  return mergeLattice;
}


const std::vector<MergeLatticePoint>& MergeLattice::getPoints() const {
  return points;
}

vector<MergeLatticePoint>
MergeLattice::getDominatedPoints(MergeLatticePoint lp) const {
  vector<MergeLatticePoint> dominatedPoints;

  // A lattice point lq is dominated by lp iff it contains a subset of lp's
  // tensor path steps. So we scan through the points and filter those points.
  vector<TensorPathStep> lpSteps = lp.getSteps();
  std::sort(lpSteps.begin(), lpSteps.end());
  for (auto& lq : getPoints()) {
    vector<TensorPathStep> lqSteps = lq.getSteps();
    std::sort(lqSteps.begin(), lqSteps.end());
    if (std::includes(lpSteps.begin(), lpSteps.end(),
                      lqSteps.begin(), lqSteps.end())) {
      dominatedPoints.push_back(lq);
    }
  }

  return dominatedPoints;
}

MergeLattice operator+(MergeLattice a, MergeLattice b) {
  vector<MergeLatticePoint> points;
  auto& aLatticePoints = a.getPoints();
  auto& bLatticePoints = b.getPoints();

  // Append all combinations of a and b lattice points
  util::append(points, (a * b).getPoints());

  // Append a lattice points
  util::append(points, aLatticePoints);

  // Append b lattice points
  util::append(points, bLatticePoints);

  return MergeLattice(points);
}

MergeLattice operator*(MergeLattice a, MergeLattice b) {
  vector<MergeLatticePoint> points;
  auto& aLatticePoints = a.getPoints();
  auto& bLatticePoints = b.getPoints();

  // Append all combinations of a and b lattice points
  for (auto& aLatticePoint : aLatticePoints) {
    for (auto& bLatticePoint : bLatticePoints) {
      points.push_back(aLatticePoint + bLatticePoint);
    }
  }

  return MergeLattice(points);
}

std::ostream& operator<<(std::ostream& os, const MergeLattice& ml) {
  return os << util::join(ml.getPoints(), " \u2228 ");
}

}}
