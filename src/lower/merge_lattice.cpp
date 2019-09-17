#include "taco/lower/merge_lattice.h"

#include <set>
#include <vector>
#include <algorithm>

#include "taco/lower/iterator.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/index_notation/index_notation_visitor.h"
#include "tensor_path.h"
#include "mode_access.h"
#include "taco/util/collections.h"
#include "taco/util/strings.h"
#include "taco/util/scopedmap.h"

using namespace std;

namespace taco {

class MergeLatticeBuilder : public IndexNotationVisitorStrict {
public:
  MergeLatticeBuilder(IndexVar i, Iterators iterators)
      : i(i), iterators(iterators) {}

  MergeLattice build(IndexStmt stmt) {
    stmt.accept(this);
    MergeLattice l = lattice;
    lattice = MergeLattice({});
    return l;
  }

  MergeLattice build(IndexExpr expr) {
    expr.accept(this);
    MergeLattice l = lattice;
    lattice = MergeLattice({});
    return l;
  }

private:
  IndexVar i;
  Iterators iterators;
  MergeLattice lattice = MergeLattice({});

  map<TensorVar,MergeLattice> latticesOfTemporaries;

  MergeLattice modeIterationLattice() {
    return MergeLattice({MergePoint({iterators.modeIterator(i)}, {}, {})});
  }

  void visit(const AccessNode* access)
  {
    if (util::contains(latticesOfTemporaries, access->tensorVar)) {
      // If the accessed tensor variable is a temporary with an associated merge
      // lattice then we return that lattice.
      lattice = latticesOfTemporaries.at(access->tensorVar);
      return;
    }

    if (!util::contains(access->indexVars,i)) {
      // The access expression does not index i so we construct a lattice from
      // the mode iterator.  This is sufficient to support broadcast semantics!
      lattice = modeIterationLattice();
      return;
    }

    Iterator iterator = getIterator(access);
    taco_iassert(iterator.hasCoordIter() || iterator.hasPosIter() ||
                 iterator.hasLocate())
        << "Iterator must support at least one capability";


    // If iterator does not support coordinate or position iteration then
    // iterate over the dimension and locate from it
    MergePoint point = (!iterator.hasCoordIter() && !iterator.hasPosIter())
                       ? MergePoint({iterators.modeIterator(i)}, {iterator}, {})
                       : MergePoint({iterator}, {}, {});

    lattice = MergeLattice({point});
  }

  void visit(const LiteralNode* node) {
    // TODO: if constant is zero, then lattice should iterate over no coordinate
    //       (rather than all coordinates)
    lattice = modeIterationLattice();
  }

  void visit(const NegNode* node) {
    lattice = build(node->a);
  }

  void visit(const AddNode* node) {
    MergeLattice a = build(node->a);
    MergeLattice b = build(node->b);
    if (a.points().size() > 0 && b.points().size() > 0) {
      lattice = unionLattices(a, b);
    }
    // Scalar operands
    else if (a.points().size() > 0) {
      lattice = a;
    }
    else if (b.points().size() > 0) {
      lattice = b;
    }
  }

  void visit(const SubNode* expr) {
    MergeLattice a = build(expr->a);
    MergeLattice b = build(expr->b);
    if (a.points().size() > 0 && b.points().size() > 0) {
      lattice = unionLattices(a, b);
    }
    // Scalar operands
    else if (a.points().size() > 0) {
      lattice = a;
    }
    else if (b.points().size() > 0) {
      lattice = b;
    }
  }

  void visit(const MulNode* expr) {
    MergeLattice a = build(expr->a);
    MergeLattice b = build(expr->b);
    if (a.points().size() > 0 && b.points().size() > 0) {
      lattice = intersectLattices(a, b);
    }
    // Scalar operands
    else if (a.points().size() > 0) {
      lattice = a;
    }
    else if (b.points().size() > 0) {
      lattice = b;
    }
  }

  void visit(const DivNode* expr) {
    MergeLattice a = build(expr->a);
    MergeLattice b = build(expr->b);
    if (a.points().size() > 0 && b.points().size() > 0) {
      lattice = intersectLattices(a, b);
    }
    // Scalar operands
    else if (a.points().size() > 0) {
      lattice = a;
    }
    else if (b.points().size() > 0) {
      lattice = b;
    }
  }

  void visit(const SqrtNode* expr) {
    lattice = build(expr->a);
  }

  void visit(const CastNode* expr) {
    lattice = build(expr->a);
  }

  void visit(const CallIntrinsicNode* expr) {
    const auto zeroPreservingArgsSets = 
        expr->func->zeroPreservingArgs(expr->args);

    std::set<size_t> zeroPreservingArgs;
    for (const auto& zeroPreservingArgsSet : zeroPreservingArgsSets) {
      taco_iassert(!zeroPreservingArgsSet.empty());
      for (const auto zeroPreservingArg : zeroPreservingArgsSet) {
        zeroPreservingArgs.insert(zeroPreservingArg);
      }
    }

    MergeLattice l = modeIterationLattice();
    for (size_t i = 0; i < expr->args.size(); ++i) {
      if (!util::contains(zeroPreservingArgs, i)) {
        MergeLattice argLattice = build(expr->args[i]);
        l = unionLattices(l, argLattice);
      }
    }

    for (const auto& zeroPreservingArgsSet : zeroPreservingArgsSets) {
      MergeLattice zeroPreservingLattice({});
      for (const auto zeroPreservingArg : zeroPreservingArgsSet) {
        MergeLattice argLattice = build(expr->args[zeroPreservingArg]);
        zeroPreservingLattice = unionLattices(zeroPreservingLattice, 
                                              argLattice);
      }
      l = intersectLattices(l, zeroPreservingLattice);
    }

    lattice = l;
  }

  void visit(const ReductionNode* node) {
    taco_ierror << "Merge lattices must be created from concrete index "
    << "notation, which does not have reduction nodes.";
  }

  void visit(const AssignmentNode* node) {
    lattice = build(node->rhs);
    latticesOfTemporaries.insert({node->lhs.getTensorVar(), lattice});

    if (util::contains(node->lhs.getIndexVars(), i)) {
      // Add result to each point in l
      Iterator result = getIterator(node->lhs);
      vector<MergePoint> points;
      for (auto& point : lattice.points()) {
        points.push_back(MergePoint(point.iterators(), point.locators(),
                                    {result}));
      }
      lattice = MergeLattice(points);
    }
  }

  void visit(const YieldNode* node) {
    lattice = build(node->expr);
  }

  void visit(const ForallNode* node) {
    lattice = build(node->stmt);
  }

  void visit(const WhereNode* node) {
    // Each where produces a temporary that is consumed on the left-hand side.
    // Since where nodes can be nested, it is possible to for multiple
    // temporaries to be consumed by a consumer expression.  The expression that
    // compute temporaries have an iteration space.  The merge lattice of these
    // iteration spaces must be merged with the iteration space of the
    // expression the temporary is combined with.  The merge lattice
    // construction strategy for where nodes is to keep a map of temporaries and
    // their corresponding merge lattices.
    build(node->producer);
    lattice = build(node->consumer);
  }

  void visit(const MultiNode* node) {
    lattice = unionLattices(build(node->stmt1), build(node->stmt2));
  }

  void visit(const SequenceNode* node) {
    taco_not_supported_yet;
  }

  Iterator getIterator(Access access) {
    taco_iassert(util::contains(access.getIndexVars(), i));
    int loc = (int)util::locate(access.getIndexVars(), i) + 1;
    return iterators.levelIterator(ModeAccess(access, loc));
  }

  /**
   * The intersection of two lattices is the result of merging all the
   * combinations of merge points from the two lattices.
   */
  static MergeLattice intersectLattices(MergeLattice left, MergeLattice right)
  {
    vector<MergePoint> points;

    // Choose a side to locate from.  We can only choose one side, we make this
    // decision once for all intersected lattice points, and we locate from the
    // right by default.
    bool locateLeft = locateFromLeft(left, right);

    // Append all combinations of a and b merge points
    for (auto& leftPoint : left.points()) {
      for (auto& rightPoint : right.points()) {
        points.push_back(intersectPoints(leftPoint, rightPoint, locateLeft));
      }
    }

    return MergeLattice(points);
  }

  /**
   * The union of two lattices is an intersection followed by the lattice
   * points of the first lattice followed by the merge points of the second.
   */
  static MergeLattice unionLattices(MergeLattice left, MergeLattice right)
  {
    vector<MergePoint> points;

    // Append all combinations of the merge points of a and b
    for (auto& apoint : left.points()) {
      for (auto& bpoint : right.points()) {
        points.push_back(unionPoints(apoint, bpoint));
      }
    }

    // Append the merge points of a
    util::append(points, left.points());

    // Append the merge points of b
    util::append(points, right.points());


    // Optimization: insert a dimension iterator if one of the iterators in the
    //               iterate set is not ordered.
    points = insertDimensionIteratorIfNotOrdered(points);

    // Optimization: move iterators to the locate set if they support locate and
    //               are subsets of some other iterator.
    points = moveLocateSubsetIteratorsToLocateSet(points);

    // Optimization: remove lattice points that lack any of the full iterators
    //               of the first point, since when a full iterator exhausts we
    //               have iterated over the whole space.
    points = removePointsThatLackFullIterators(points);

    // Optimization: remove lattice points whose iterators are identical to the
    //               iterators of an earlier point, since we have already iterated
    //               over this sub-space.
    points = removePointsWithIdenticalIterators(points);

    return MergeLattice(points);
  }

  /**
   * Conjunctively merge two merge points a and b into a new point. The steps
   * of the new merge point are a union (concatenation) of the steps of a and
   * b. The expression of the new merge point is expr_a op expr_b, where op is
   * a binary expr type.  If locateLeft is true then we will locate from a,
   *  otherwise we will locate from b.
   */
  static MergePoint intersectPoints(MergePoint left, MergePoint right,
                                    bool locateLeft)
  {
    vector<Iterator> iterators;
    vector<Iterator> locators;

    tie(iterators, locators) = split((locateLeft ? left : right).iterators(),
                                     [](Iterator it){return !it.hasLocate();});
    iterators = filter(iterators, [](Iterator it) {
      return !it.isDimensionIterator();
    });

    iterators = (locateLeft) ? combine(iterators, right.iterators())
                             : combine(left.iterators(), iterators);
    locators = (locateLeft) ? combine(locators, left.locators(),
                                      right.locators())
                            : combine(left.locators(), locators,
                                      right.locators());

    // Remove duplicate iterators.
    iterators = deduplicateDimensionIterators(iterators);

    vector<Iterator> results = combine(left.results(),   right.results());

    return MergePoint(iterators, locators, results);
  }

  /**
   * Disjunctively merge two merge points a and b into a new point. The steps
   * of the new merge point are a union (concatenation) of the steps of a and
   * b. The expression of the new merge point is expr_a op expr_b, where op is
   * a binary expr type.
   */
  static MergePoint unionPoints(MergePoint left, MergePoint right)
  {
    vector<Iterator> iterators= combine(left.iterators(),right.iterators());
    vector<Iterator> locaters = combine(left.locators(), right.locators());
    vector<Iterator> results  = combine(left.results(),  right.results());

    // Remove duplicate iterators.
    iterators = deduplicateDimensionIterators(iterators);

    return MergePoint(iterators, locaters, results);
  }

  static bool locateFromLeft(MergeLattice left, MergeLattice right)
  {
    // Locate from the side with a dimension iterator
    if (any(right.iterators(),
            [](Iterator it){ return it.isDimensionIterator(); })) {
      return false;
    }
    if (any(left.iterators(),
            [](Iterator it){ return it.isDimensionIterator(); })) {
      return true;
    }

    // Locate from the side with a full+locate iterator
    if (any(right.iterators(),
            [](Iterator it){ return it.isFull() && it.hasLocate(); })) {
      return false;
    }
    if (any(left.iterators(),
            [](Iterator it){ return it.isFull() && it.hasLocate(); })) {
      return true;
    }

    // Locate from the side with more locate iterators
    size_t leftNumLocates  = count(left.iterators(),
                                   [](Iterator it){ return it.hasLocate(); });
    size_t rightNumLocates = count(right.iterators(),
                                   [](Iterator it){ return it.hasLocate(); });
    return (leftNumLocates > rightNumLocates);
  }

  static vector<MergePoint>
  insertDimensionIteratorIfNotOrdered(vector<MergePoint> points)
  {
    vector<MergePoint> results;
    for (auto& point : points) {
      vector<Iterator> iterators = point.iterators();
      if (any(iterators, [](Iterator it){ return !it.isOrdered(); }) &&
          !any(iterators, [](Iterator it){ return it.isDimensionIterator(); })) {
        taco_iassert(point.iterators().size() > 0);
        Iterator dimension(iterators[0].getIndexVar());
        results.push_back(MergePoint(combine(iterators, {dimension}),
                                     point.locators(),
                                     point.results()));
      }
      else {
        results.push_back(point);
      }
    }
    return results;
  }

  static vector<MergePoint>
  moveLocateSubsetIteratorsToLocateSet(vector<MergePoint> points)
  {
    vector<Iterator> full = filter(points[0].iterators(),
                                   [](Iterator it){ return it.isFull(); });

    // We only support, for now, optimizing for subsets of full iterators.  If
    // there are no full iterators then we don't do anything.
    if (full.size() == 0) {
      return points;
    }

    // Move locate iterators to the locate set, except the first full iterator.
    Iterator firstFull = full[0];
    vector<MergePoint> result;
    for (auto& point : points) {
      vector<Iterator> locators;
      vector<Iterator> iterators;
      tie(locators, iterators) = split(point.iterators(),
                                       [&firstFull](Iterator it) {
                                         return it.hasLocate() && it != firstFull;
                                       });
      result.push_back(MergePoint(iterators,
                                  combine(point.locators(), locators),
                                  point.results()));
    }
    return result;
  }

  static vector<MergePoint>
  removePointsThatLackFullIterators(vector<MergePoint> points)
  {
    vector<MergePoint> result;
    vector<Iterator> fullIterators = filter(points[0].iterators(),
                                            [](Iterator it){return it.isFull();});
    for (auto& point : points) {
      bool missingFullIterator = false;
      for (auto& fullIterator : fullIterators) {
        if (!util::contains(point.iterators(), fullIterator)) {
          missingFullIterator = true;
          break;
        }
      }
      if (!missingFullIterator) {
        result.push_back(point);
      }
    }
    return result;
  }

  static vector<MergePoint>
  removePointsWithIdenticalIterators(vector<MergePoint> points)
  {
    vector<MergePoint> result;
    set<set<Iterator>> iteratorSets;
    for (auto& point : points) {
      set<Iterator> iteratorSet(point.iterators().begin(), 
                                point.iterators().end());
      if (util::contains(iteratorSets, iteratorSet)) {
        continue;
      }
      result.push_back(point);
      iteratorSets.insert(iteratorSet);
    }
    return result;
  }

  static vector<Iterator>
  deduplicateDimensionIterators(const vector<Iterator>& iterators)
  {
    vector<Iterator> deduplicates;

    // Remove all but one of the dense iterators, which are all the same.
    bool dimensionIteratorFound = false;
    for (auto& iterator : iterators) {
      if (iterator.isDimensionIterator()) {
        if (!dimensionIteratorFound) {
          deduplicates.push_back(iterator);
          dimensionIteratorFound = true;
        }
      }
      else {
        deduplicates.push_back(iterator);
      }
    }
    return deduplicates;
  }
};


// class MergeLattice
MergeLattice::MergeLattice(vector<MergePoint> points) : points_(points)
{
}

MergeLattice MergeLattice::make(Forall forall, Iterators iterators)
{
  MergeLatticeBuilder builder(forall.getIndexVar(), iterators);
  return builder.build(forall.getStmt());
}

MergeLattice MergeLattice::subLattice(MergePoint lp) const {
  // A merge point lp dominats lq iff it contains a subset of lp's
  // tensor path steps. So we scan through the points and filter those points.
  vector<MergePoint> dominatedPoints;
  vector<Iterator> lpIterators = lp.iterators();
  sort(lpIterators.begin(), lpIterators.end());
  for (auto& lq : this->points()) {
    vector<Iterator> lqIterators = lq.iterators();
    sort(lqIterators.begin(), lqIterators.end());
    if (includes(lpIterators.begin(), lpIterators.end(),
                      lqIterators.begin(), lqIterators.end())) {
      dominatedPoints.push_back(lq);
    }
  }
  return MergeLattice(dominatedPoints);
}

const vector<MergePoint>& MergeLattice::points() const {
  return points_;
}

const vector<Iterator>& MergeLattice::iterators() const {
  // The iterators merged by a lattice are those merged by the first point
  taco_iassert(points().size() > 0) << "No merge points in the merge lattice";
  return points()[0].iterators();
}

set<Iterator> MergeLattice::exhausted(MergePoint point) {
  set<Iterator> notExhaustedIters(point.iterators().begin(),
                                  point.iterators().end());

  set<Iterator> exhausted;
  for (auto& iterator : iterators()) {
    if (!util::contains(notExhaustedIters, iterator)) {
      exhausted.insert(iterator);
    }
  }

  return exhausted;
}

const std::vector<Iterator>& MergeLattice::results() const {
  taco_iassert(points().size() > 0) << "No merge points in the merge lattice";
  return points()[0].results();
}

bool MergeLattice::exact() const {
  // A lattice is full if any merge point iterates over only full iterators
  // or if each sparse iterator is uniquely iterated by some lattice point.
  set<Iterator> uniquelyMergedIterators;
  for (auto& point : this->points()) {
    if (all(point.iterators(), [](Iterator it) {return it.isFull();})) {
      return true;
    }
  }

  for (auto& point : this->points()) {
    if (point.iterators().size() == 1) {
      uniquelyMergedIterators.insert(point.iterators()[0]);
    }
  }

  for (auto& it : iterators()) {
    if (!util::contains(uniquelyMergedIterators, it)) {
      return false;
    }
  }
  return true;
}

ostream& operator<<(ostream& os, const MergeLattice& ml) {
  return os << util::join(ml.points(), ", ");
}

bool operator==(const MergeLattice& a, const MergeLattice& b) {
  auto& apoints = a.points();
  auto& bpoints = b.points();
  if (apoints.size() != bpoints.size()) {
    return false;
  }
  for (size_t i = 0; i < apoints.size(); i++) {
    if (apoints[i] != bpoints[i]) {
      return false;
    }
  }
  return true;
}

bool operator!=(const MergeLattice& a, const MergeLattice& b) {
  return !(a == b);
}


// class MergePoint
struct MergePoint::Content {
  std::vector<Iterator> iterators;
  std::vector<Iterator> locators;
  std::vector<Iterator> results;
};

MergePoint::MergePoint(const vector<Iterator>& iterators,
                       const vector<Iterator>& locators,
                       const vector<Iterator>& results) : content_(new Content) {
  taco_uassert(all(iterators,
                   [](Iterator it){ return it.hasLocate() || it.isOrdered(); }))
      << "Merge points do not support iterators that do not have locate and "
      << "that are not ordered.";

  content_->iterators = iterators;
  content_->locators = locators;
  content_->results = results;
}

const vector<Iterator>& MergePoint::iterators() const {
  return content_->iterators;
}

std::vector<Iterator> MergePoint::rangers() const {
  // We can remove an iterator from the rangers iff it is guaranteed to be
  // exhausted after the other rangers (the rangers are the iterators we iterate
  // over until one is exhausted).  This holds if the largest coordinate of this
  // iterator is smaller than the largest coordinate of the other iterators.
  // We will, conservatively, say this condition holds if the iterator is full
  // and there exist another iterator that is not full, since this iterator is
  // then a superset of that iterator.  We will start with all iterators and
  // only add those for which this condition does not hold to the rangers.
  if (any(iterators(), [](Iterator iterator){return !iterator.isFull();})) {
    vector<Iterator> rangers;
    for (auto& iterator : iterators()) {
      if (!iterator.isFull()) {
        rangers.push_back(iterator);
      }
    }
    return rangers;
  }
  return iterators();
}

std::vector<Iterator> MergePoint::mergers() const {
  // We can remove an iterator from the mergers iff it is a subset of the other
  // mergers (the mergers ar the iterators that specify the points we visit
  // within the range specified by the rangers).  We will, conservatively, say
  // that this condition holds if one of the other iterators is full, since this
  // iterator is then a subset of it.  We will start with all iterators and only
  // add those for which this condition does not hold to the mergers.
  if (any(iterators(), [](Iterator iterator){return iterator.isFull();})) {
    vector<Iterator> mergers;
    for (auto& iterator : iterators()) {
      if (iterator.isFull()) {
        mergers.push_back(iterator);
      }
    }
    return mergers;
  }
  return iterators();
}

const std::vector<Iterator>& MergePoint::locators() const {
  return content_->locators;
}

const std::vector<Iterator>& MergePoint::results() const {
  return content_->results;
}

ostream& operator<<(ostream& os, const MergePoint& mlp) {
  os << "[";
  os << util::join(mlp.iterators(), ", ");
  if (mlp.iterators().size() > 0) os << " ";
  os << "|";
  os << " ";
  os << util::join(mlp.locators(),  ", ");
  if (mlp.locators().size() > 0) os << " ";
  os << "|";
  if (mlp.results().size() > 0) os << " ";
  os << util::join(mlp.results(),   ", ");
  os << "]";
  return os;
}

static bool compare(const vector<Iterator>& a, const vector<Iterator>& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (auto iterators : util::zip(a,b)) {
    if (iterators.first != iterators.second) {
      return false;
    }
  }
  return true;
}

bool operator==(const MergePoint& a, const MergePoint& b) {
  if (!compare(a.iterators(), b.iterators())) return false;
  if (!compare(a.locators(), b.locators())) return false;
  if (!compare(a.results(), b.results())) return false;
  return true;
}

bool operator!=(const MergePoint& a, const MergePoint& b) {
  return !(a == b);
}


// Free functions
std::vector<Iterator> deduplicate(const std::vector<Iterator>& iterators) {
  vector<Iterator> deduplicates;

  // Remove all but one of the dense iterators, which are all the same.
  bool added = false;
  for (auto& iterator : iterators) {
    if (iterator.isFull() && iterator.isOrdered()) {
      if (!added) {
        deduplicates.push_back(iterator);
        added = true;
      }
    }
    else {
      deduplicates.push_back(iterator);
    }
  }
  return deduplicates;
}

vector<Iterator> simplify(const vector<Iterator>& iterators) {
  vector<Iterator> simplifiedIterators;
  vector<Iterator> fullIterators;

  for (const auto& iter : iterators) {
    if (!iter.isFull()) {
      simplifiedIterators.push_back(iter);
    } else if (fullIterators.empty()) {
      // must iterate over at least one of the full dimensions
      fullIterators.push_back(iter);
    } else if (!iter.hasLocate()) {
      // preferably iterate over only full dimensions that do not support locate
      if (fullIterators[0].hasLocate()) {
        fullIterators.clear();
      }
      fullIterators.push_back(iter);
    }
  }
  util::append(simplifiedIterators, fullIterators);

  return simplifiedIterators;
}

}
