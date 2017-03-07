#ifndef TACO_MERGE_LATTICE_H
#define TACO_MERGE_LATTICE_H

#include <ostream>
#include <vector>

#include "expr.h"
#include "storage/iterator.h"

namespace taco {
class Var;

namespace lower {

class IterationSchedule;
class TensorPathStep;
class MergeLatticePoint;
class Iterators;


/// A merge lattice represents a sequence of disjunctions, where each term is a
/// MergeLatticePoint.
class MergeLattice {
public:
  MergeLattice();
  MergeLattice(std::vector<MergeLatticePoint> points);

  /// Constructs a merge lattice for an index expression and an index variable.
  static MergeLattice make(const Expr& indexExpr, const Var& indexVar,
                           const IterationSchedule& schedule,
                           const Iterators& iterators);

  /// Returns the lattice points of this merge lattice.
  const std::vector<MergeLatticePoint>& getPoints() const;

  /// Returns the steps merged by this merge lattice.
  const std::vector<TensorPathStep>& getSteps() const;

  /// Returns all the iterators that are merged by this lattice
  const std::vector<storage::Iterator>& getIterators() const;

  /// Returns the expression merged by the lattice.
  const Expr& getExpr() const;

  /// Returns the lattice points in this merge lattice that are (non-strictly)
  /// dominated by lp.
  std::vector<MergeLatticePoint> getDominatedPoints(MergeLatticePoint lp) const;

  /// Returns true if the merge lattice has any lattice points, false otherwise.
  bool defined() const;

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

/// Compare two merge lattices
bool operator==(const MergeLattice&, const MergeLattice&);
bool operator!=(const MergeLattice&, const MergeLattice&);



/// A merge lattice point, which represents a conjunction of tensor paths.
class MergeLatticePoint {
public:
  MergeLatticePoint(std::vector<TensorPathStep> steps, const Expr& expr,
                    const std::vector<storage::Iterator>& iterators);

    /// Returns the operand tensor path steps merged by this lattice point.
  const std::vector<TensorPathStep>& getSteps() const;

  // Removes redundant steps from the lattice point to simplify the merge.
  // This means removing dense steps since these are supersets of sparse steps
  // and since $S \intersect D = S$. If there are no sparse steps then the
  // simplified lattice point consist of a single dense step.
  MergeLatticePoint simplify();

  /// Returns the iterators that are merged by this lattice point
  const std::vector<storage::Iterator>& getIterators() const;

  /// Returns the expression merged by the lattice point.
  const Expr& getExpr() const;

private:
  std::vector<TensorPathStep> steps;

  Expr expr;
  std::vector<storage::Iterator> iterators;
};

/// Merge two lattice points a and b into a new point. The steps of the new
/// lattice point are a union (concatenation) of the steps of a and b. The
/// expression of the new lattice point is expr_a op expr_b, where op is a
/// binary expr type.
template<class op>
MergeLatticePoint merge(MergeLatticePoint a, MergeLatticePoint b);

/// Print a merge lattice point
std::ostream& operator<<(std::ostream&, const MergeLatticePoint&);

/// Compare two merge lattice points
bool operator==(const MergeLatticePoint&, const MergeLatticePoint&);
bool operator!=(const MergeLatticePoint&, const MergeLatticePoint&);

}}
#endif
