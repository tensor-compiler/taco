#ifndef TACO_MERGE_LATTICE_H
#define TACO_MERGE_LATTICE_H

#include <ostream>
#include <vector>

namespace taco {
namespace is {
class TensorPath;
class MergeRule;


/// A merge lattice point, which represents a conjunction of tensor paths.
class MergeLatticePoint {
public:
  MergeLatticePoint(const TensorPath& path);

  const std::vector<TensorPath>& getPaths() const;

  friend MergeLatticePoint operator+(MergeLatticePoint, MergeLatticePoint);

private:
  std::vector<TensorPath> paths;

  MergeLatticePoint(std::vector<TensorPath> paths);
};

std::ostream& operator<<(std::ostream&, const MergeLatticePoint&);


/// A merge lattice, which represents a sequence of disjunctions, where each
/// term is a MergeLatticePoint.
class MergeLattice {
public:
  MergeLattice();
  MergeLattice(MergeLatticePoint point);

  const std::vector<MergeLatticePoint>& getPoints() const;

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
