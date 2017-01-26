#ifndef TACO_MERGE_LATTICE_H
#define TACO_MERGE_LATTICE_H

#include <ostream>
#include <vector>

#include "expr.h"

namespace taco {
namespace lower {

class IterationSchedule;
class TensorPathStep;
class MergeRule;

/// A merge lattice point, which represents a conjunction of tensor paths.
class MergeLatticePoint {
public:
  MergeLatticePoint(const TensorPathStep& step);
  MergeLatticePoint(std::vector<TensorPathStep> steps);

  const std::vector<TensorPathStep>& getSteps() const;

  // Removes redundant steps from the lattice point to simplify the merge.
  // This means removing dense steps since these are supersets of sparse steps
  // and since $S \intersect D = S$. If there are no sparse steps then the
  // simplified lattice point consist of a single dense step.
  MergeLatticePoint simplify();

  friend MergeLatticePoint operator+(MergeLatticePoint, MergeLatticePoint);

private:
  std::vector<TensorPathStep> steps;
};

std::ostream& operator<<(std::ostream&, const MergeLatticePoint&);


/// A merge lattice, which represents a sequence of disjunctions, where each
/// term is a MergeLatticePoint.
class MergeLattice {
public:
  MergeLattice();

  /// Build a merge lattice from a merge rule
  static MergeLattice make(const MergeRule& rule);

  const std::vector<MergeLatticePoint>& getPoints() const;

  /// Returns the lattice points in this merge lattice that are (non-strictly)
  /// dominated by lp.
  std::vector<MergeLatticePoint> getDominatedPoints(MergeLatticePoint lp) const;

  friend MergeLattice operator+(MergeLattice, MergeLattice);
  friend MergeLattice operator*(MergeLattice, MergeLattice);

private:
  std::vector<MergeLatticePoint> points;

  MergeLattice(std::vector<MergeLatticePoint> points);
  MergeLattice(MergeLatticePoint point);
};

std::ostream& operator<<(std::ostream&, const MergeLattice&);

}}
#endif
