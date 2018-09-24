#ifndef TACO_MERGE_LATTICE_OLD_H
#define TACO_MERGE_LATTICE_OLD_H

#include <ostream>
#include <vector>

#include "taco/lower/iterator.h"
#include "taco/index_notation/index_notation.h"

namespace taco {

class ModeAccess;
class IndexVar;

namespace old {
class IterationGraph;
class Iterators;

class MergePoint;

/// A merge lattice represents a sequence of disjunctions, where each term is a
/// MergeLatticePoint.
class MergeLattice {
public:
  MergeLattice();

  /// Construct a merge lattice containing the given points.
  MergeLattice(std::vector<MergePoint> points,
               std::vector<Iterator> resultIterators);

  /// Construct a merge lattice from a forall statement.
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
  const std::vector<MergePoint>& getPoints() const;

  /// Retrieve the iterators merged by this lattice.
  const std::vector<Iterator>& getIterators() const;

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

/// A merge point represent the iteration over the intersection of the sparse
/// iteration spaces of one or more iterators.  A merge point provides five sets
/// of iterators that are used in different ways:
/// - Rangers are the iterators that must be co-iterated to cover the sparse
///   iteration space.
/// - Mergers are the iterators whose coordinates must be merged (with min) to
///   compute the coordinate of each point in the sparse iteration space.
/// - Locaters are the iterators whose coordinates must be retrieved through
///   their locate capability.
/// - Appenders are the result iterators that are appended to.
/// - Inserters are the result iterators that are inserted into.
class MergePoint {
public:
  MergePoint(std::vector<Iterator> iterators, std::vector<Iterator> rangeIters,
             std::vector<Iterator> mergeIters, IndexExpr expr);

  /// Returns all the iterators of this merge point. These are the iterators
  /// that may be accessed in each iteration of the merge point loop.
  const std::vector<Iterator>& getIterators() const;

  /// Returns the iterators that must be coiterated. These exclude full
  /// iterators that support locate.
  const std::vector<Iterator>& getRangers() const;

  /// Returns the subset of iterators that must be merged to cover the points
  /// of the iteration space of this merge lattice. These exclude iterators that
  /// support locate.
  const std::vector<Iterator>& getMergers() const;

  /// Returns the expression merged by the merge point.
  const IndexExpr& getExpr() const;

private:
  std::vector<Iterator> iterators;
  std::vector<Iterator> mergers;
  std::vector<Iterator> rangers;
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

}}
#endif
