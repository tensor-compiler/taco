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
class MergeLatticePoint;


/// A merge lattice represents a sequence of disjunctions, where each term is a
/// MergeLatticePoint.
class MergeLattice {
public:
  MergeLattice();
  MergeLattice(std::vector<MergeLatticePoint> points);
  MergeLattice(MergeLatticePoint point);

  /// Build a merge lattice from a merge rule
  static MergeLattice make(const MergeRule& rule);

  /// Returns the lattice points of this merge lattice.
  const std::vector<MergeLatticePoint>& getPoints() const;

  /// Reurns the steps merged by this merge lattice.
  const std::vector<TensorPathStep>& getSteps() const;

  /// Returns the expression merged by the lattice.
  const Expr& getExpr() const;

  /// Returns the lattice points in this merge lattice that are (non-strictly)
  /// dominated by lp.
  std::vector<MergeLatticePoint> getDominatedPoints(MergeLatticePoint lp) const;

private:
  std::vector<MergeLatticePoint> points;
};

/// The conjunction of two lattices is the result of merging all the
/// combinations of lattice points from the two lattices. The expression of the
/// new lattice is expr_a op expr_b, where op is a binary expr type.
template<class op>
MergeLattice conjunction(MergeLattice a, MergeLattice b);

/// The disjunction of two lattices is a conjunction followed by the lattice
/// points of the first lattice followed by the lattice points of the second.
/// The expression of the new lattice is expr_a op expr_b, where op is a binary
/// expr type.
template<class op>
MergeLattice disjunction(MergeLattice a, MergeLattice b);

/// Print a merge lattice
std::ostream& operator<<(std::ostream&, const MergeLattice&);


/// A merge lattice point, which represents a conjunction of tensor paths.
class MergeLatticePoint {
public:
  MergeLatticePoint(const TensorPathStep& step, const Expr& expr);
  MergeLatticePoint(std::vector<TensorPathStep> steps, const Expr& expr);

    /// Returns the operand tensor path steps merged by this lattice point.
  const std::vector<TensorPathStep>& getSteps() const;

  // Removes redundant steps from the lattice point to simplify the merge.
  // This means removing dense steps since these are supersets of sparse steps
  // and since $S \intersect D = S$. If there are no sparse steps then the
  // simplified lattice point consist of a single dense step.
  MergeLatticePoint simplify();

  /// Returns the expression merged by the lattice point.
  const Expr& getExpr() const;

private:
  std::vector<TensorPathStep> steps;
  Expr expr;
};

/// Merge two lattice points a and b into a new point. The steps of the new
/// lattice point are a union (concatenation) of the steps of a and b. The
/// expression of the new lattice point is expr_a op expr_b, where op is a
/// binary expr type.
template<class op>
MergeLatticePoint merge(MergeLatticePoint a, MergeLatticePoint b);

/// Print a merge lattice point
std::ostream& operator<<(std::ostream&, const MergeLatticePoint&);

}}
#endif
