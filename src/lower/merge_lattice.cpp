#include "merge_lattice.h"

#include <set>
#include <algorithm>

#include "taco/expr_nodes/expr_nodes.h"
#include "taco/expr_nodes/expr_visitor.h"
#include "iteration_schedule.h"
#include "iterators.h"
#include "taco/util/collections.h"
#include "taco/util/strings.h"

using namespace std;
using namespace taco::expr_nodes;

namespace taco {
namespace lower {

// class MergeLattice
MergeLattice::MergeLattice() {
}

MergeLattice::MergeLattice(vector<MergeLatticePoint> points) : points(points){
}

template <class op>
static
MergeLattice scale(MergeLattice lattice, IndexExpr scale, bool leftScale) {
  vector<MergeLatticePoint> scaledPoints;
  for (auto& point : lattice) {
    IndexExpr expr = point.getExpr();
    IndexExpr scaledExpr = (leftScale) ? new op(scale, expr)
                                       : new op(expr, scale);
    MergeLatticePoint scaledPoint(point.getIterators(),
                                  point.getMergeIterators(), scaledExpr);
    scaledPoints.push_back(scaledPoint);
  }
  return MergeLattice(scaledPoints);
}

template <class op>
static MergeLattice scale(IndexExpr expr, MergeLattice lattice) {
  return scale<op>(lattice, expr, true);
}

template <class op>
static MergeLattice scale(MergeLattice lattice, IndexExpr expr) {
  return scale<op>(lattice, expr, false);
}

template <class op>
static MergeLattice unary(MergeLattice lattice) {
  vector<MergeLatticePoint> negPoints;
  for (auto& point : lattice) {
    IndexExpr negExpr = new op(point.getExpr());
    negPoints.push_back(MergeLatticePoint(point.getIterators(),
                                          point.getMergeIterators(), negExpr));
  }
  return MergeLattice(negPoints);
}

MergeLattice MergeLattice::make(const IndexExpr& indexExpr,
                                const IndexVar& indexVar,
                                const IterationSchedule& schedule,
                                const Iterators& iterators) {
  struct BuildMergeLattice : public expr_nodes::ExprVisitorStrict {
    const IndexVar&          indexVar;
    const IterationSchedule& schedule;
    const Iterators&         iterators;
    MergeLattice             lattice;

    BuildMergeLattice(const IndexVar& indexVar,
                      const IterationSchedule& schedule,
                      const Iterators& iterators)
        : indexVar(indexVar), schedule(schedule), iterators(iterators) {
    }

    MergeLattice buildLattice(const IndexExpr& expr) {
      expr.accept(this);
      MergeLattice l = lattice;
      lattice = MergeLattice();
      return l;
    }

    using ExprVisitorStrict::visit;

    void visit(const AccessNode* expr) {
      // Throw away expressions `var` does not contribute to
      if (!util::contains(expr->indexVars, indexVar)) {
        lattice = MergeLattice();
        return;
      }

      TensorPath path = schedule.getTensorPath(expr);
      size_t i = util::locate(path.getVariables(), indexVar);
      storage::Iterator iter = iterators[path.getStep(i)];
      MergeLatticePoint latticePoint = MergeLatticePoint({iter}, {iter}, expr);
      lattice = MergeLattice({latticePoint});
    }

    void visit(const NegNode* expr) {
      MergeLattice a = buildLattice(expr->a);
      lattice = unary<NegNode>(a);
    }

    void visit(const SqrtNode* expr) {
      MergeLattice a = buildLattice(expr->a);
      lattice = unary<SqrtNode>(a);
    }

    void visit(const AddNode* expr) {
      MergeLattice a = buildLattice(expr->a);
      MergeLattice b = buildLattice(expr->b);
      if (a.defined() && b.defined()) {
        lattice = disjunction<AddNode>(a, b);
      }
      // Scalar operands
      else if (a.defined()) {
        lattice = scale<AddNode>(a, expr->b);
      }
      else if (b.defined()) {
        lattice = scale<AddNode>(expr->a, b);
      }
    }

    void visit(const SubNode* expr) {
      MergeLattice a = buildLattice(expr->a);
      MergeLattice b = buildLattice(expr->b);
      if (a.defined() && b.defined()) {
        lattice = disjunction<SubNode>(a, b);
      }
      // Scalar operands
      else if (a.defined()) {
        lattice = scale<SubNode>(a, expr->b);
      }
      else if (b.defined()) {
        lattice = scale<SubNode>(expr->a, b);
      }
    }

    void visit(const MulNode* expr) {
      MergeLattice a = buildLattice(expr->a);
      MergeLattice b = buildLattice(expr->b);
      if (a.defined() && b.defined()) {
        lattice = conjunction<MulNode>(a, b);
      }
      // Scalar operands
      else if (a.defined()) {
        lattice = scale<MulNode>(a, expr->b);
      }
      else if (b.defined()) {
        lattice = scale<MulNode>(expr->a, b);
      }
    }

    void visit(const DivNode* expr) {
      MergeLattice a = buildLattice(expr->a);
      MergeLattice b = buildLattice(expr->b);
      if (a.defined() && b.defined()) {
        lattice = conjunction<DivNode>(a, b);
      }
      // Scalar operands
      else if (a.defined()) {
        lattice = scale<DivNode>(a, expr->b);
      }
      else if (b.defined()) {
        lattice = scale<DivNode>(expr->a, b);
      }
    }

    void visit(const IntImmNode*) {}
    void visit(const FloatImmNode*) {}
    void visit(const DoubleImmNode*) {}
  };

  auto lattice =
      BuildMergeLattice(indexVar, schedule, iterators).buildLattice(indexExpr);
  taco_iassert(lattice.getSize() > 0) <<
      "Every merge lattice should have at least one lattice point";
  return lattice;
}

size_t MergeLattice::getSize() const {
  return points.size();
}

const MergeLatticePoint& MergeLattice::operator[](size_t i) const {
  return points[i];
}

const std::vector<storage::Iterator>& MergeLattice::getIterators() const {
  // The iterators merged by a lattice are those merged by the first point
  taco_iassert(points.size() > 0) << "No lattice points in the merge lattice";
  return points[0].getIterators();
}

const IndexExpr& MergeLattice::getExpr() const {
  taco_iassert(points.size() > 0) << "No lattice points in the merge lattice";

  // The expression merged by a lattice is the same as the expression of the
  // first lattice point
  return points[0].getExpr();
}

MergeLattice MergeLattice::getSubLattice(MergeLatticePoint lp) const {
  // A lattice point lp dominats lq iff it contains a subset of lp's
  // tensor path steps. So we scan through the points and filter those points.
  vector<MergeLatticePoint> dominatedPoints;
  vector<storage::Iterator> lpIterators = lp.getIterators();
  std::sort(lpIterators.begin(), lpIterators.end());
  for (auto& lq : *this) {
    vector<storage::Iterator> lqIterators = lq.getIterators();
    std::sort(lqIterators.begin(), lqIterators.end());
    if (std::includes(lpIterators.begin(), lpIterators.end(),
                      lqIterators.begin(), lqIterators.end())) {
      dominatedPoints.push_back(lq);
    }
  }
  return MergeLattice(dominatedPoints);
}

bool MergeLattice::isFull() const {
  // A merge lattice is full if any lattice point merges a single dense iterator
  // or if each sparse iterator is uniquely merged by some lattice point.
  set<storage::Iterator> uniquelyMergedIterators;
  for (auto& point : *this) {
    if (point.getMergeIterators().size()== 1 ) {
      auto it = point.getMergeIterators()[0];
      uniquelyMergedIterators.insert(it);
      if (it.isDense()) {
        return true;
      }
    }
  }
  for (auto& it : getIterators()) {
    if (!util::contains(uniquelyMergedIterators, it)) {
      return false;
    }
  }
  return true;
}

bool MergeLattice::defined() const {
  return points.size() > 0;
}

std::vector<MergeLatticePoint>::iterator MergeLattice::begin() {
  return points.begin();
}


std::vector<MergeLatticePoint>::iterator MergeLattice::end() {
  return points.end();
}

std::vector<MergeLatticePoint>::const_iterator MergeLattice::begin() const {
  return points.begin();
}


std::vector<MergeLatticePoint>::const_iterator MergeLattice::end() const {
  return points.end();
}

template<class op>
MergeLattice conjunction(MergeLattice a, MergeLattice b) {
  vector<MergeLatticePoint> points;

  // Append all combinations of a and b lattice points
  for (auto& aLatticePoint : a) {
    for (auto& bLatticePoint : b) {
      points.push_back(conjunction<op>(aLatticePoint, bLatticePoint));
    }
  }

  return MergeLattice(points);
}

template<class op>
MergeLattice disjunction(MergeLattice a, MergeLattice b) {
  vector<MergeLatticePoint> points;

  // Append all combinations of the lattice points of a and b
  vector<MergeLatticePoint> allPoints;
  for (auto& aLatticePoint : a) {
    for (auto& bLatticePoint : b) {
      allPoints.push_back(disjunction<op>(aLatticePoint, bLatticePoint));
    }
  }

  // Append the lattice points of a
  util::append(allPoints, a);

  // Append the lattice points of b
  util::append(allPoints, b);

  taco_iassert(allPoints.size() > 0) <<
      "A lattice must have at least one point";

  // Exhausting a dense iterator cause the lattice to drop to zero. Therefore
  // we cannot end up in a lattice point that doesn't contain the dense iterator
  // and must remove all lattice points that don't contain it.
  auto denseIterators = getDenseIterators(allPoints[0].getIterators());
  for (auto& point : allPoints) {
    bool missingDenseIterator = false;
    for (auto& denseIterator : denseIterators) {
      if (!util::contains(point.getIterators(), denseIterator)) {
        missingDenseIterator = true;
        break;
      }
    }
    if (!missingDenseIterator) {
      points.push_back(point);
    }
  }

  MergeLattice lattice = MergeLattice(points);
  taco_iassert(lattice.getSize() > 0) <<
      "All lattices must have at least one point";
  return lattice;
}

std::ostream& operator<<(std::ostream& os, const MergeLattice& ml) {
  return os << util::join(ml, " \u2228 ");
}

bool operator==(const MergeLattice& a, const MergeLattice& b) {
  if (a.getSize() != b.getSize()) {
    return false;
  }
  for (size_t i = 0; i < a.getSize(); i++) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

bool operator!=(const MergeLattice& a, const MergeLattice& b) {
  return !(a == b);
}


// class MergeLatticePoint
MergeLatticePoint::MergeLatticePoint(vector<storage::Iterator> iterators,
                                     IndexExpr expr)
    : iterators(iterators), rangeIterators(simplify(iterators)), expr(expr) {
}

MergeLatticePoint::MergeLatticePoint(vector<storage::Iterator> iterators,
                                     vector<storage::Iterator> mergeIterators,
                                     IndexExpr expr)
    : iterators(iterators), rangeIterators(simplify(iterators)),
      mergeIterators(mergeIterators), expr(expr) {
}

const vector<storage::Iterator>& MergeLatticePoint::getIterators() const {
  return iterators;
}

const vector<storage::Iterator>& MergeLatticePoint::getRangeIterators() const {
  return rangeIterators;
}

const vector<storage::Iterator>& MergeLatticePoint::getMergeIterators() const {
  return mergeIterators;
}

const IndexExpr& MergeLatticePoint::getExpr() const {
  return expr;
}

template<class op>
MergeLatticePoint merge(MergeLatticePoint a, MergeLatticePoint b,
                        bool conjunctive) {
  vector<storage::Iterator> iters;
  iters.insert(iters.end(), a.getIterators().begin(), a.getIterators().end());
  iters.insert(iters.end(), b.getIterators().begin(), b.getIterators().end());

  IndexExpr expr = new op(a.getExpr(), b.getExpr());

  vector<storage::Iterator> mergeIters;
  auto& aMergeIters = a.getMergeIterators();
  auto& bMergeIters = b.getMergeIterators();

  // A merge iterator list consists of either one dense or n sparse iterators.
  taco_iassert(aMergeIters.size() >= 0 && (aMergeIters.size() == 1 ||
               getDenseIterators(aMergeIters).size() == 0));
  taco_iassert(bMergeIters.size() >= 0 && (bMergeIters.size() == 1 ||
               getDenseIterators(bMergeIters).size() == 0));

  // If both merge iterator lists consist of sparse iterators then the result
  // is a union of those lists
  if (!aMergeIters[0].isDense() && !bMergeIters[0].isDense()) {
    mergeIters.insert(mergeIters.end(), aMergeIters.begin(), aMergeIters.end());
    mergeIters.insert(mergeIters.end(), bMergeIters.begin(), bMergeIters.end());
  }
  // If both merge iterator lists consist of a dense iterator then the result
  // is a dense iterator
  else if (aMergeIters[0].isDense() && bMergeIters[0].isDense()) {
    mergeIters = aMergeIters;
  }
  // If one merge iterator list consist of a dense iterators and the other
  // consist of sparse iterators
  else {
    // Conjunctive operator: the result is the list of sparse iterators
    if (conjunctive) {
      mergeIters =  aMergeIters[0].isDense() ? bMergeIters : aMergeIters;
    }
    // Disjunctive operator: the result is the dense iterator
    else {
      mergeIters =  aMergeIters[0].isDense() ? aMergeIters : bMergeIters;
    }
  }
  taco_iassert(mergeIters.size() > 0);

  return MergeLatticePoint(iters,  mergeIters, expr);
}

template<class op>
MergeLatticePoint conjunction(MergeLatticePoint a, MergeLatticePoint b) {
  return merge<op>(a, b, true);
}

template<class op>
MergeLatticePoint disjunction(MergeLatticePoint a, MergeLatticePoint b) {
  return merge<op>(a, b, false);
}

std::ostream& operator<<(std::ostream& os, const MergeLatticePoint& mlp) {
  vector<string> pathNames;
  os << "[";
  os << util::join(mlp.getIterators(), " \u2227 ");
  os << "]";
  return os;
}

bool operator==(const MergeLatticePoint& a, const MergeLatticePoint& b) {
  auto& aiters = a.getIterators();
  auto& biters = b.getIterators();
  if (aiters.size() != biters.size()) {
    return false;
  }
  for (size_t i = 0; i < aiters.size(); i++) {
    if (aiters[i] != biters[i]) {
      return false;
    }
  }
  return true;
}

bool operator!=(const MergeLatticePoint& a, const MergeLatticePoint& b) {
  return !(a == b);
}

vector<storage::Iterator> simplify(const vector<storage::Iterator>& iterators) {
  vector<storage::Iterator> simplifiedIterators;

  // Remove dense iterators
  for (size_t i = 0; i < iterators.size(); i++) {
    auto iter = iterators[i];
    if (!iter.isDense()) {
      simplifiedIterators.push_back(iter);
    }
  }

  // If there are only dense iterators then keep the first one
  if (simplifiedIterators.size() == 0) {
    taco_iassert(iterators.size() > 0);
    simplifiedIterators.push_back(iterators[0]);
  }

  return simplifiedIterators;
}

}}
