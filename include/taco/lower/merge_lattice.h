#ifndef TACO_MERGE_LATTICE_H
#define TACO_MERGE_LATTICE_H

#include <vector>
#include <set>
#include <map>
#include <ostream>
#include <memory>
#include "taco/index_notation/index_notation.h"

namespace taco {

class Iterator;
class Iterators;
class Forall;
class ModeAccess;
class MergePoint;

/**
 * A merge lattice represents a sequence of disjunctions, where each term is a
 * MergeLatticePoint.
 */
class MergeLattice {
public:
  /**
   * Construct a merge lattice from a concrete index notation forall statement
   * that describes how to iterate through the sparse iteration space of the
   * forall statement index variable's sparse iteration space.
   *
   * \param forall
   *      A forall concrete index notation statement to construct a lattice for.
   *
   * \param iterators
   *      Iterators that iterate over the tensor coordinate hierarchies of every
   *      access expression and that iterate over tensor modes.
   *
   * \param provGraph
   *      Tracks relationships between index variables in concrete notation
   *
   * \param definedIndexVars
   *      The set of index variables that appear in enclosing loops and therefore have
   *      a concrete value within the context of this merge lattice
   *
   * \param whereTempsToResult
   *      Maps between temporary tensors and where their result will be stored necessary for when
   *      we use scalar temporaries inside of a loop (to reduce the number of atomic instructions for example),
   *      but we still need to know what the result location for this temporary is so that it can be properly coiterated
   */
  static MergeLattice make(Forall forall, Iterators iterators, ProvenanceGraph provGraph,
          std::set<IndexVar> definedIndexVars, std::map<TensorVar, const AccessNode *> whereTempsToResult = {});


  /**
   * Removes lattice points whose iterators are identical to the iterators of an earlier point, since we have
   * already iterated over this sub-space.
   */
  static std::vector<MergePoint> removePointsWithIdenticalIterators(const std::vector<MergePoint>&);

  /**
   * remove lattice points that lack any of the full iterators of the first point, since when a full iterator exhausts
   * we have iterated over the whole space.
   */
  static std::vector<MergePoint> removePointsThatLackFullIterators(const std::vector<MergePoint>&);

  /// Returns true if we need to emit checks for explicit zeros in the lattice given.
  bool needExplicitZeroChecks();

  /**
   * Returns the sub-lattice rooted at the given merge point.
   */
  MergeLattice subLattice(MergePoint lp) const;

  /**
   * Retrieve the merge points.
   */
  const std::vector<MergePoint>& points() const;

  /**
   * Retrieve all the iterators merged by this lattice.
   */
  const std::vector<Iterator>& iterators() const;

  /**
 * Retrieve all the locators in this lattice.
 */
  const std::vector<Iterator>& locators() const;

  /**
   * Returns iterators that have been exhausted prior to the merge point.
   */
  std::set<Iterator> exhausted(MergePoint point);

  /**
   * Retrieve all the results written to in this merge lattice.
   */
  const std::vector<Iterator>& results() const;

  /**
   * True if the merge lattice enumerates the iteration space exactly, meaning
   * no point in the space will be considered and discarded.
   */
  bool exact() const;

  /**
   * True if any of the mode iterators in the lattice is a leaf iterator.
   *
   * This method checks both the iterators and locators since they both contain mode iterators.
   */
  bool anyModeIteratorIsLeaf() const;

  /**
   * Get a list of iterators that should be omitted at this merge point.
   */
  std::vector<Iterator> retrieveRegionIteratorsToOmit(const MergePoint& point) const;

  /**
   * Returns a set of sets of tensor iterators. A merge point with a tensorRegion in this set should not
   * be removed from the lattice.
   *
   * Needed so that special regions are kept when applying optimizations that remove merge points.
   */
   std::set<std::set<Iterator>> getTensorRegionsToKeep() const;

  /**
    * Removes points from the lattice that would duplicate iteration over the input tensors.
    */
  MergeLattice getLoopLattice() const;

private:
  std::vector<MergePoint> points_;
  std::set<std::set<Iterator>> regionsToKeep;

public:
  /**
   * Construct a merge lattice containing the given points.  This constructor
   * is primarily intended for testing purposes and most construction should
   * happen through `MergeLattice::make`.
   */
  MergeLattice(std::vector<MergePoint> points, std::set<std::set<Iterator>> regionsToKeep = {});
};

std::ostream& operator<<(std::ostream&, const MergeLattice&);
bool operator==(const MergeLattice&, const MergeLattice&);
bool operator!=(const MergeLattice&, const MergeLattice&);


/**
 *  A merge point represent iteration until an intersection of the sparse
 *  iteration spaces is exhausted.  A merge point divides iterators into three
 *  sets that are used in different ways:
 *
 *  - Iterators are iterated co-iterated to visit the coordinates of this point.
 *    Iteraters are further classified as rangers and mergers (an iterator can
 *    be both a ranger and a merger).  The rangers are those iterators we must
 *    iterate until exhaustion while the mergers are those whose coordinates
 *    we must merge with the min function to get the candidate coordinate.
 *    We can always treat all iterators as both rangers mergers; however, when
 *    some iterators are supersets of the rest, we can optimize by removing
 *    iterators from the two sets.
 *
 *  - Locaters are queried with coordinates to retrieve positions, and must
 *    therefore support the locate capability.
 *
 *  - Results are appended to or inserted into.
 */
class MergePoint {
public:
  /**
   * Returns the iterators that co-iterate over this merge point.
   */
  const std::vector<Iterator>& iterators() const;

  /**
   * Retrieve the ranger iterators, which are those iterators we must iterate
   * over to exhaustion.
   */
  std::vector<Iterator> rangers() const;

  /**
   * Retrieve the merger iterators, which are those iterators whose coordinates
   * must be merged with the min function to get the candidate coordinate.
   */
  std::vector<Iterator> mergers() const;

  /**
   * Retrieve the iterators whose positions must be computed using the locate
   * function and the resolved coordinate.
   */
  const std::vector<Iterator>& locators() const;

  /**
   * Retrieve the iterators of the results to be appended to or inserted into.
   */
  const std::vector<Iterator>& results() const;

  /**
   * Returns the iterators that iterate over or locate into tensors
   */
  const std::set<Iterator> tensorRegion() const;

  /**
   * Returns true if this merge point may leave out the tensors it iterates
   */
   bool isOmitter() const;

private:
  struct Content;
  std::shared_ptr<Content> content_;

public:
  /**
   * Construct a merge point.  This constructor is primarily intended for
   * testing purposes and most construction should happen through
   * `MergeLattice::make`.
   */
  MergePoint(const std::vector<Iterator>& iterators,
             const std::vector<Iterator>& locators,
             const std::vector<Iterator>& results,
             bool omitPoint = false);
};

std::ostream& operator<<(std::ostream&, const MergePoint&);
bool operator==(const MergePoint&, const MergePoint&);
bool operator!=(const MergePoint&, const MergePoint&);

/**
 * Remove coordinate iterators that iterate over the same coordinates, such
 * as full ordered coordinate iterators.
 */
std::vector<Iterator> deduplicate(const std::vector<Iterator>& iterators);

/**
 * Simplify a set of iterators by removing redundant iterators. This means
 * removing dense iterators since these are supersets of sparse iterators and
 * since $S \intersect D = S$. If there are no sparse steps then the simplified
 * merge point consist of a single dense step.
 */
std::vector<Iterator> simplify(const std::vector<Iterator>&);

}
#endif
