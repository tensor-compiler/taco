#include "merge_lattice.h"

#include <set>
#include <algorithm>

#include "taco/operator.h"
#include "taco/expr_nodes/expr_nodes.h"
#include "taco/expr_nodes/expr_visitor.h"
#include "iteration_schedule.h"
#include "iterators.h"
#include "taco/util/collections.h"
#include "taco/util/strings.h"

using namespace std;

namespace taco {
namespace lower {

// class MergeLattice
MergeLattice::MergeLattice() {
}

MergeLattice::MergeLattice(vector<MergeLatticePoint> points) : points(points) {
}

template <class op>
static MergeLattice scale(MergeLattice lattice, Expr scale, bool leftScale) {
  vector<MergeLatticePoint> scaledPoints;
  for (auto& point : lattice) {
    Expr expr = point.getExpr();
    Expr scaledExpr = (leftScale) ? op(scale, expr)
                                  : op(expr, scale);
    MergeLatticePoint scaledPoint(point.getIterators(), scaledExpr);
    scaledPoints.push_back(scaledPoint);
  }
  return MergeLattice(scaledPoints);
}

template <class op>
static MergeLattice scale(Expr expr, MergeLattice lattice) {
  return scale<op>(lattice, expr, true);
}

template <class op>
static MergeLattice scale(MergeLattice lattice, Expr expr) {
  return scale<op>(lattice, expr, false);
}

template <class op>
static MergeLattice unary(MergeLattice lattice) {
  vector<MergeLatticePoint> negPoints;
  for (auto& point : lattice) {
    Expr negExpr = op(point.getExpr());
    negPoints.push_back(MergeLatticePoint(point.getIterators(), negExpr));
  }
  return MergeLattice(negPoints);
}

MergeLattice MergeLattice::make(const Expr& indexExpr, const Var& indexVar,
                                const IterationSchedule& schedule,
                                const Iterators& iterators) {
  struct BuildMergeLattice : public internal::ExprVisitorStrict {
    const Var&               indexVar;
    const IterationSchedule& schedule;
    const Iterators&         iterators;
    MergeLattice             lattice;

    BuildMergeLattice(const Var& indexVar, const IterationSchedule& schedule,
                      const Iterators& iterators)
        : indexVar(indexVar), schedule(schedule), iterators(iterators) {
    }

    MergeLattice buildLattice(const Expr& expr) {
      expr.accept(this);
      MergeLattice l = lattice;
      lattice = MergeLattice();
      return l;
    }

    using ExprVisitorStrict::visit;

    void visit(const internal::Read* expr) {
      // Throw away expressions `var` does not contribute to
      if (!util::contains(expr->indexVars, indexVar)) {
        lattice = MergeLattice();
        return;
      }

      TensorPath path = schedule.getTensorPath(expr);
      size_t i = util::locate(path.getVariables(), indexVar);
      vector<TensorPathStep> steps = {path.getStep(i)};
      auto latticePoint = MergeLatticePoint(iterators[steps], expr);
      lattice = MergeLattice({latticePoint});
    }

    void visit(const internal::Neg* expr) {
      MergeLattice a = buildLattice(expr->a);
      lattice = unary<Neg>(a);
    }

    void visit(const internal::Sqrt* expr) {
      MergeLattice a = buildLattice(expr->a);
      lattice = unary<Sqrt>(a);
    }

    void visit(const internal::Add* expr) {
      MergeLattice a = buildLattice(expr->a);
      MergeLattice b = buildLattice(expr->b);
      if (a.defined() && b.defined()) {
        lattice = disjunction<Add>(a, b);
      }
      // Scalar operands
      else if (a.defined()) {
        lattice = scale<Add>(a, expr->b);
      }
      else if (b.defined()) {
        lattice = scale<Add>(expr->a, b);
      }
    }

    void visit(const internal::Sub* expr) {
      MergeLattice a = buildLattice(expr->a);
      MergeLattice b = buildLattice(expr->b);
      if (a.defined() && b.defined()) {
        lattice = disjunction<Sub>(a, b);
      }
      // Scalar operands
      else if (a.defined()) {
        lattice = scale<Sub>(a, expr->b);
      }
      else if (b.defined()) {
        lattice = scale<Sub>(expr->a, b);
      }
    }

    void visit(const internal::Mul* expr) {
      MergeLattice a = buildLattice(expr->a);
      MergeLattice b = buildLattice(expr->b);
      if (a.defined() && b.defined()) {
        lattice = conjunction<Mul>(a, b);
      }
      // Scalar operands
      else if (a.defined()) {
        lattice = scale<Mul>(a, expr->b);
      }
      else if (b.defined()) {
        lattice = scale<Mul>(expr->a, b);
      }
    }

    void visit(const internal::Div* expr) {
      MergeLattice a = buildLattice(expr->a);
      MergeLattice b = buildLattice(expr->b);
      if (a.defined() && b.defined()) {
        lattice = conjunction<Div>(a, b);
      }
      // Scalar operands
      else if (a.defined()) {
        lattice = scale<Div>(a, expr->b);
      }
      else if (b.defined()) {
        lattice = scale<Div>(expr->a, b);
      }
    }

    void visit(const internal::IntImm*) {
      not_supported_yet;
    }

    void visit(const internal::FloatImm*) {
      not_supported_yet;
    }

    void visit(const internal::DoubleImm*) {
      not_supported_yet;
    }
  };

  auto lattice =
      BuildMergeLattice(indexVar, schedule, iterators).buildLattice(indexExpr);
  iassert(lattice.getSize() > 0) <<
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
  iassert(points.size() > 0) << "No lattice points in the merge lattice";
  return points[0].getIterators();
}

const Expr& MergeLattice::getExpr() const {
  iassert(points.size() > 0) << "No lattice points in the merge lattice";

  // The expression merged by a lattice is the same as the expression of the
  // first lattice point
  return points[0].getExpr();
}

vector<MergeLatticePoint>
MergeLattice::getDominatedPoints(MergeLatticePoint lp) const {
  vector<MergeLatticePoint> dominatedPoints;

  // A lattice point lq is dominated by lp iff it contains a subset of lp's
  // tensor path steps. So we scan through the points and filter those points.
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
  return dominatedPoints;
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
      points.push_back(merge<op>(aLatticePoint, bLatticePoint));
    }
  }

  return MergeLattice(points);
}

template<class op>
MergeLattice disjunction(MergeLattice a, MergeLattice b) {
  vector<MergeLatticePoint> allPoints;

  // Append all combinations of the lattice points of a and b
  util::append(allPoints, conjunction<op>(a,b));

  // Append the lattice points of a
  util::append(allPoints, a);

  // Append the lattice points of b
  util::append(allPoints, b);

  iassert(allPoints.size() > 0) << "A lattice must have at least one point";

  // Exhausting a dense iterator cause the lattice to drop to zero. Therefore
  // we cannot end up in a lattice point that doesn't contain the dense iterator
  // and must remove all lattice points that don't contain it.
  // TODO: Technically we should remove points dominated by a point with an
  //       iterator that is a non-strict superset of all the other iterators in,
  //       and not just for dense iterators which are superset of all other
  //       iterators.
  auto denseIterators = getDenseIterators(allPoints[0].getIterators());
  vector<MergeLatticePoint> points;
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
  iassert(lattice.getSize() > 0) << "All lattices must have at least one point";
  return lattice;
}

std::ostream& operator<<(std::ostream& os, const MergeLattice& ml) {
  return os << util::join(ml, "  \u2228  ");
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
                                     Expr expr)
    : iterators(iterators), expr(expr) {
}

const vector<storage::Iterator>& MergeLatticePoint::getIterators() const {
  return iterators;
}

const Expr& MergeLatticePoint::getExpr() const {
  return expr;
}

template<class op>
MergeLatticePoint merge(MergeLatticePoint a, MergeLatticePoint b) {
  vector<storage::Iterator> iters;
  iters.insert(iters.end(), a.getIterators().begin(), a.getIterators().end());
  iters.insert(iters.end(), b.getIterators().begin(), b.getIterators().end());

  Expr expr = op(a.getExpr(), b.getExpr());
  return MergeLatticePoint(iters, expr);
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
    iassert(iterators.size() > 0);
    simplifiedIterators.push_back(iterators[0]);
  }

  return simplifiedIterators;
}

}}
