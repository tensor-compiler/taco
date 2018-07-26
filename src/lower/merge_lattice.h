#ifndef TACO_MERGE_LATTICE_H
#define TACO_MERGE_LATTICE_H

#include <ostream>
#include <vector>

#include "iterator.h"
#include "taco/index_notation/index_notation.h"

namespace taco {

class ModeAccess;
class IndexVar;
class MergePoint;

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
  MergeLattice(std::vector<MergePoint> points,
               std::vector<Iterator> resultIterators);

  /// Construct a merge lattice f
  static MergeLattice make(Forall forall,
                           const std::map<ModeAccess,Iterator>& iterators);

  /// Constructs a merge lattice for an index expression and an index variable.
  /// @deprecated
  static MergeLattice make(const IndexExpr& indexExpr,
                           const IndexVar& indexVar,
                           const old::IterationGraph& iterationGraph,
                           const old::Iterators& iterators);

  /// Returns the number of merge points in this lattice
  size_t getSize() const;

  /// Retrieve the ith merge point of this merge lattice.
  const MergePoint& operator[](size_t i) const;

  /// Retrieve the merge points.
  const std::vector<MergePoint>& getMergePoints() const;

  /// Retrieve the iterators that are merged by this lattice.
  const std::vector<Iterator>& getMergeIterators() const;

  /// Retrieve the iterators that must be coiterated.
  const std::vector<Iterator>& getRangeIterators() const;

  /// Retrieve the result iterators.
  const std::vector<Iterator>& getResultIterators() const;

  /// Returns the expression merged by the lattice.
  const IndexExpr& getExpr() const;

  /// Returns the sub-lattice rooted at the given merge point.
  MergeLattice getSubLattice(MergePoint lp) const;

  /// True if the merge lattice enumerates the whole iteration space, which
  /// means that no point in the space will be considered and discarded.
  bool isFull() const;

  /// Returns true if the merge lattice has any merge points, false otherwise.
  bool defined() const;

  /// Iterator to the first merge point
  std::vector<MergePoint>::iterator begin();

  /// Iterator past the last merge point
  std::vector<MergePoint>::iterator end();

  /// Iterator to the first merge point
  std::vector<MergePoint>::const_iterator begin() const;

  /// Iterator past the last merge point
  std::vector<MergePoint>::const_iterator end() const;

private:
  std::vector<MergePoint> mergePoint;
  std::vector<Iterator> resultIterators;
};

/// The intersection of two lattices is the result of merging all the
/// combinations of merge points from the two lattices. The expression of the
/// new lattice is expr_a op expr_b, where op is a binary expr type.
template<class op>
MergeLattice mergeIntersection(MergeLattice a, MergeLattice b);

/// The union of two lattices is an intersection followed by the lattice
/// points of the first lattice followed by the merge points of the second.
/// The expression of the new lattice is expr_a op expr_b, where op is a binary
/// expr type.
template<class op>
MergeLattice mergeUnion(MergeLattice a, MergeLattice b);

/// Print a merge lattice
std::ostream& operator<<(std::ostream&, const MergeLattice&);

/// Compare two merge lattices
bool operator==(const MergeLattice&, const MergeLattice&);
bool operator!=(const MergeLattice&, const MergeLattice&);


/// A merge point represents iterating over a sparse space until the
/// intersection intersection between some of the iterators has been exhausted.
class MergePoint {
public:
  MergePoint(std::vector<Iterator> iterators,
             std::vector<Iterator> mergeIters,
             std::vector<Iterator> rangeIters,
             IndexExpr expr);

  /// Returns all the iterators of this merge point. These are the iterators
  /// that may be accessed in each iteration of the merge point loop.
  const std::vector<Iterator>& getIterators() const;

  /// Returns the subset of iterators that needs to be explicitly merged to 
  /// cover the points of the iteration space of this merge lattice. These 
  /// exclude iterators that can be accessed with locate.
  const std::vector<Iterator>& getMergeIterators() const;

  /// Returns the iterators that need to be explicitly coiterated in order to be 
  /// merged. These exclude iterators over full dimensions that support locate.
  const std::vector<Iterator>& getRangeIterators() const;

  /// Returns the expression merged by the merge point.
  const IndexExpr& getExpr() const;

private:
  std::vector<Iterator> iterators;
  std::vector<Iterator> mergeIterators;
  std::vector<Iterator> rangeIterators;
  IndexExpr expr;
};

/// Conjunctively merge two merge points a and b into a new point. The steps
/// of the new merge point are a union (concatenation) of the steps of a and
/// b. The expression of the new merge point is expr_a op expr_b, where op is
/// a binary expr type.
template<class op>
MergePoint mergeIntersection(MergePoint a, MergePoint b);

/// Disjunctively merge two merge points a and b into a new point. The steps
/// of the new merge point are a union (concatenation) of the steps of a and
/// b. The expression of the new merge point is expr_a op expr_b, where op is
/// a binary expr type.
template<class op>
MergePoint mergeUnion(MergePoint a, MergePoint b);


/// Print a merge point
std::ostream& operator<<(std::ostream&, const MergePoint&);

/// Compare two merge points
bool operator==(const MergePoint&, const MergePoint&);
bool operator!=(const MergePoint&, const MergePoint&);

/// Simplify iterators by removing redundant iterators. This means removing
/// dense iterators since these are supersets of sparse iterators and since
/// $S \intersect D = S$. If there are no sparse steps then the simplified
/// merge point consist of a single dense step.
std::vector<Iterator> simplify(const std::vector<Iterator>&);

/// Returns the Access expressions that have become exhausted prior to the
/// merge point in the lattice.
std::set<Access> exhaustedAccesses(MergePoint, MergeLattice);

}
#endif
