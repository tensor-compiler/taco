#ifndef TACO_MERGE_LATTICE_H
#define TACO_MERGE_LATTICE_H

#include <ostream>
#include <vector>

namespace taco {
namespace lower {
class TensorPathStep;
class MergeRule;


/// A merge lattice point, which represents a conjunction of tensor paths.
class MergeLatticePoint {
public:
  MergeLatticePoint(const TensorPathStep& step);

  const std::vector<TensorPathStep>& getSteps() const;

  friend MergeLatticePoint operator+(MergeLatticePoint, MergeLatticePoint);

private:
  std::vector<TensorPathStep> steps;

  MergeLatticePoint(std::vector<TensorPathStep> steps);
};

std::ostream& operator<<(std::ostream&, const MergeLatticePoint&);


/// A merge lattice, which represents a sequence of disjunctions, where each
/// term is a MergeLatticePoint.
class MergeLattice {
public:
  MergeLattice();
  MergeLattice(MergeLatticePoint point);

  const std::vector<MergeLatticePoint>& getPoints() const;

  /// Returns the lattice points that are (non-strictly) dominated by the
  /// given lattice point.
  std::vector<MergeLatticePoint> getDominatedPoints(MergeLatticePoint) const;

  friend MergeLattice operator+(MergeLattice, MergeLattice);
  friend MergeLattice operator*(MergeLattice, MergeLattice);

private:
  std::vector<MergeLatticePoint> points;

  MergeLattice(std::vector<MergeLatticePoint> points);
};

std::ostream& operator<<(std::ostream&, const MergeLattice&);


/// Build a merge lattice from a merge rule
MergeLattice buildMergeLattice(const MergeRule& rule);

}}
#endif
