#ifndef TACO_MERGE_LATTICE_H
#define TACO_MERGE_LATTICE_H

#include <ostream>
#include <vector>

#include "taco/index_notation/index_notation.h"
#include "storage/iterator.h"

namespace taco {

class IndexVar;
class MergeLatticePoint;

namespace old {
class IterationGraph;
class Iterators;
}

/// A merge lattice represents a sequence of disjunctions, where each term is a
/// MergeLatticePoint.
class MergeLattice {
public:
  MergeLattice();

  /// Construct a merge lattice containing the given points.
  MergeLattice(std::vector<MergeLatticePoint> points);

  /// Constructs a merge lattice for an index expression and an index variable.
  static MergeLattice make(const IndexExpr& indexExpr,
                           const IndexVar& indexVar,
                           const old::IterationGraph& iterationGraph,
                           const old::Iterators& iterators);

  /// Returns the number of lattice points in this lattice
  size_t getSize() const;

  /// Returns the ith lattice point of this merge lattice.
  const MergeLatticePoint& operator[](size_t i) const;

  /// Returns all the iterators that are merged by this lattice.
  const std::vector<Iterator>& getIterators() const;

  /// Returns all the iterators that must be coiterated.
  const std::vector<Iterator>& getRangeIterators() const;

  /// Returns the expression merged by the lattice.
  const IndexExpr& getExpr() const;

  /// Returns the sub-lattice rooted at the given lattice point.
  MergeLattice getSubLattice(MergeLatticePoint lp) const;

  /// True if the merge lattice enumerates the whole iteration space, which
  /// means that no point in the space will be considered and discarded.
  bool isFull() const;

  /// Returns true if the merge lattice has any lattice points, false otherwise.
  bool defined() const;

  /// Iterator to the first lattice point
  std::vector<MergeLatticePoint>::iterator begin();

  /// Iterator past the last lattice point
  std::vector<MergeLatticePoint>::iterator end();

  /// Iterator to the first lattice point
  std::vector<MergeLatticePoint>::const_iterator begin() const;

  /// Iterator past the last lattice point
  std::vector<MergeLatticePoint>::const_iterator end() const;

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
  MergeLatticePoint(std::vector<Iterator> iterators,
                    std::vector<Iterator> mergeIters,
                    std::vector<Iterator> rangeIters,
                    IndexExpr expr);

  /// Returns all the iterators of this lattice point. These are the iterators
  /// that may be accessed in each iteration of the lattice point loop.
  const std::vector<Iterator>& getIterators() const;

  /// Returns the subset of iterators that needs to be explicitly merged to 
  /// cover the points of the iteration space of this merge lattice. These 
  /// exclude iterators that can be accessed with locate.
  const std::vector<Iterator>& getMergeIterators() const;

  /// Returns the iterators that need to be explicitly coiterated in order to be 
  /// merged. These exclude iterators over full dimensions that support locate.
  const std::vector<Iterator>& getRangeIterators() const;

  /// Returns the expression merged by the lattice point.
  const IndexExpr& getExpr() const;

private:
  std::vector<Iterator> iterators;
  std::vector<Iterator> mergeIterators;
  std::vector<Iterator> rangeIterators;
  IndexExpr expr;
};

/// Conjunctively merge two lattice points a and b into a new point. The steps
/// of the new lattice point are a union (concatenation) of the steps of a and
/// b. The expression of the new lattice point is expr_a op expr_b, where op is
/// a binary expr type.
template<class op>
MergeLatticePoint conjunction(MergeLatticePoint a, MergeLatticePoint b);

/// Disjunctively merge two lattice points a and b into a new point. The steps
/// of the new lattice point are a union (concatenation) of the steps of a and
/// b. The expression of the new lattice point is expr_a op expr_b, where op is
/// a binary expr type.
template<class op>
MergeLatticePoint disjunction(MergeLatticePoint a, MergeLatticePoint b);


/// Print a merge lattice point
std::ostream& operator<<(std::ostream&, const MergeLatticePoint&);

/// Compare two merge lattice points
bool operator==(const MergeLatticePoint&, const MergeLatticePoint&);
bool operator!=(const MergeLatticePoint&, const MergeLatticePoint&);

/// Simplify iterators by removing redundant iterators. This means removing
/// dense iterators since these are supersets of sparse iterators and since
/// $S \intersect D = S$. If there are no sparse steps then the simplified
/// lattice point consist of a single dense step.
std::vector<Iterator> simplify(const std::vector<Iterator>&);

/// Returns the Access expressions that have become exhausted prior to the
/// lattice point in the lattice.
std::set<Access> exhaustedAccesses(MergeLatticePoint, MergeLattice);

}
#endif
