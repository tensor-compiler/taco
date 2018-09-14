#ifndef TACO_MERGE_LATTICE_OLD_H
#define TACO_MERGE_LATTICE_OLD_H

#include <ostream>
#include <vector>
#include <set>

#include "taco/lower/iterator.h"

namespace taco {

class ModeAccess;
class Forall;
class Access;
class MergePoint;

/// A merge lattice represents a sequence of disjunctions, where each term is a
/// MergeLatticePoint.
class MergeLattice {
public:
  /// Construct a merge lattice containing the given points.
  MergeLattice(std::vector<MergePoint> points);

  /// Construct a merge lattice from a forall statement.
  static MergeLattice make(Forall forall,
                           const std::map<ModeAccess,Iterator>& iterators);

  /// Returns the sub-lattice rooted at the given merge point.
  MergeLattice subLattice(MergePoint lp) const;

  /// Retrieve the merge points.
  const std::vector<MergePoint>& getPoints() const;

  /// Retrieve the iterators merged by this lattice.
  const std::vector<Iterator>& getIterators() const;

   /// Retrieve the iterators that must be co-iterated.  This means we must
  /// iterate until one of them is exhausted.
  const std::vector<Iterator>& getRangers() const;

  /// Retrieve the results written to in this merge lattice.
  const std::vector<Iterator>& getResults() const;

  /// True if the merge lattice enumerates the whole iteration space, which
  /// means that no point in the space will be considered and discarded.
  bool isFull() const;

  /// Returns true if the merge lattice has any merge points, false otherwise.
  bool defined() const;

private:
  std::vector<MergePoint> points;
};

/// The intersection of two lattices is the result of merging all the
/// combinations of merge points from the two lattices. The expression of the
/// new lattice is expr_a op expr_b, where op is a binary expr type.
MergeLattice latticeIntersection(MergeLattice a, MergeLattice b);

/// The union of two lattices is an intersection followed by the lattice
/// points of the first lattice followed by the merge points of the second.
/// The expression of the new lattice is expr_a op expr_b, where op is a binary
/// expr type.
MergeLattice latticeUnion(MergeLattice a, MergeLattice b);

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
/// - Results are the iterators that are appended to or inserted into.
class MergePoint {
public:
  /// Construct a merge point.
  MergePoint(const std::vector<Iterator>& iterators,
             const std::vector<Iterator>& rangers,
             const std::vector<Iterator>& mergers,
             const std::vector<Iterator>& locaters,
             const std::vector<Iterator>& results);

  /// Returns the iterators that co-iterate over this merge point.
  const std::vector<Iterator>& getIterators() const;

  /// Retrieve the iterators that must be co-iterated.  This means we must
  /// iterate until one of them is exhausted.
  const std::vector<Iterator>& getRangers() const;

  /// Retrieve the iterators whose candidate coordinates must be combined
  /// (with min) to compute the resolved coordinate.
  const std::vector<Iterator>& getMergers() const;

  /// Retrieve the iterators whose positions must be computed using the locate
  /// function and the resolved coordinate.
  const std::vector<Iterator>& getLocators() const;

  /// Retrieve the results written to in this merge point.
  const std::vector<Iterator>& getResults() const;

private:
  struct Content;
  std::shared_ptr<Content> content;
};

/// Split iterators into rangers and mergers.  The rangers are those we must
/// iterate over until exhaustion while the mergers are those whose coordinates
/// we must merge with the min function to get the candidate coordinate.  This
/// is an optimization and we can always treat all iterators as both rangers
/// and mergers.  When some iterators are supersets of the rest, however, we
/// can range over the subset and merge the coordinates from the superset.
std::pair<std::vector<Iterator>, std::vector<Iterator>>
splitRangersAndMergers(const std::vector<Iterator>& iterators);

/// Remove coordinate iterators that iterate over the same coordinates, such
/// as full ordered coordinate iterators.
std::vector<Iterator> deduplicate(const std::vector<Iterator>& iterators);

/// Conjunctively merge two merge points a and b into a new point. The steps
/// of the new merge point are a union (concatenation) of the steps of a and
/// b. The expression of the new merge point is expr_a op expr_b, where op is
/// a binary expr type.
MergePoint pointIntersection(MergePoint a, MergePoint b);

/// Disjunctively merge two merge points a and b into a new point. The steps
/// of the new merge point are a union (concatenation) of the steps of a and
/// b. The expression of the new merge point is expr_a op expr_b, where op is
/// a binary expr type.
MergePoint pointUnion(MergePoint a, MergePoint b);

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
