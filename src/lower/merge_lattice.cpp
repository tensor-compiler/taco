#include "merge_lattice.h"

#include <algorithm>

#include "expr_nodes.h"
#include "expr_visitor.h"
#include "operator.h"

#include "internal_tensor.h"
#include "iteration_schedule.h"
#include "tensor_path.h"
#include "util/collections.h"
#include "util/strings.h"

using namespace std;

namespace taco {
namespace lower {

// class MergeLattice
MergeLattice::MergeLattice() {
}


MergeLattice::MergeLattice(vector<MergeLatticePoint> points)
    : points(points) {
}

template <class op>
static MergeLattice scale(MergeLattice lattice, Expr scale, bool leftScale) {
  auto& points = lattice.getPoints();
  vector<MergeLatticePoint> scaledPoints;
  for (auto& point : points) {
    Expr expr = point.getExpr();
    Expr scaledExpr = (leftScale) ? op(scale, expr)
                                  : op(expr, scale);
    MergeLatticePoint scaledPoint(point.getSteps(),scaledExpr);
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
  auto& points = lattice.getPoints();
  vector<MergeLatticePoint> negPoints;
  for (auto& point : points) {
    Expr negExpr = op(point.getExpr());
    negPoints.push_back(MergeLatticePoint(point.getSteps(), negExpr));
  }
  return MergeLattice(negPoints);
}

MergeLattice MergeLattice::make(const Expr& indexExpr, const Var& indexVar,
                                const IterationSchedule& schedule) {
  struct BuildMergeLattice : public internal::ExprVisitorStrict {
    const Var&               indexVar;
    const IterationSchedule& schedule;
    MergeLattice             lattice;

    BuildMergeLattice(const Var& indexVar, const IterationSchedule& schedule)
        : indexVar(indexVar), schedule(schedule) {
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
      lattice = MergeLattice({MergeLatticePoint({path.getStep(i)}, expr)});
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

  return BuildMergeLattice(indexVar,schedule).buildLattice(indexExpr);
}

const std::vector<MergeLatticePoint>& MergeLattice::getPoints() const {
  return points;
}

const std::vector<TensorPathStep>& MergeLattice::getSteps() const {
  iassert(points.size() > 0) << "No lattice points in the merge lattice";

  // The steps merged by a lattice is the same as the expression of the
  // first lattice point
  return points[0].getSteps();
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
  vector<TensorPathStep> lpSteps = lp.getSteps();
  std::sort(lpSteps.begin(), lpSteps.end());
  for (auto& lq : getPoints()) {
    vector<TensorPathStep> lqSteps = lq.getSteps();
    std::sort(lqSteps.begin(), lqSteps.end());
    if (std::includes(lpSteps.begin(), lpSteps.end(),
                      lqSteps.begin(), lqSteps.end())) {
      dominatedPoints.push_back(lq);
    }
  }
  return dominatedPoints;
}

bool MergeLattice::defined() const {
  return points.size() > 0;
}

template<class op>
MergeLattice conjunction(MergeLattice a, MergeLattice b) {
  vector<MergeLatticePoint> points;
  auto& aLatticePoints = a.getPoints();
  auto& bLatticePoints = b.getPoints();

  // Append all combinations of a and b lattice points
  for (auto& aLatticePoint : aLatticePoints) {
    for (auto& bLatticePoint : bLatticePoints) {
      points.push_back(merge<op>(aLatticePoint, bLatticePoint));
    }
  }

  return MergeLattice(points);
}

template<class op>
MergeLattice disjunction(MergeLattice a, MergeLattice b) {
  vector<MergeLatticePoint> points;
  auto& aLatticePoints = a.getPoints();
  auto& bLatticePoints = b.getPoints();

  // Append all combinations of a and b lattice points
  util::append(points, conjunction<op>(a,b).getPoints());

  // Append a lattice points
  util::append(points, aLatticePoints);

  // Append b lattice points
  util::append(points, bLatticePoints);

  return MergeLattice(points);
}

std::ostream& operator<<(std::ostream& os, const MergeLattice& ml) {
  return os << util::join(ml.getPoints(), " \u2228 ");
}

bool operator==(const MergeLattice& a, const MergeLattice& b) {
  auto& apoints = a.getPoints();
  auto& bpoints = b.getPoints();
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



// class MergeLatticePoint
MergeLatticePoint::MergeLatticePoint(std::vector<TensorPathStep> steps,
                                     const Expr& expr)
    : steps(steps), expr(expr) {
}

const std::vector<TensorPathStep>& MergeLatticePoint::getSteps() const {
  return steps;
}

MergeLatticePoint MergeLatticePoint::simplify() {
  vector<TensorPathStep> steps;

  // Remove dense steps
  for (auto& step : getSteps()) {
    Format format = step.getPath().getTensor().getFormat();
    if (format.getLevels()[step.getStep()].getType() != LevelType::Dense) {
      steps.push_back(step);
    }
  }

  // If there are only dense steps then keep the first
  if (steps.size() == 0) {
    iassert(getSteps().size() > 0);
    steps.push_back(getSteps()[0]);
  }

  return MergeLatticePoint(steps, this->getExpr());
}

const Expr& MergeLatticePoint::getExpr() const {
  return expr;
}

template<class op>
MergeLatticePoint merge(MergeLatticePoint a, MergeLatticePoint b) {
  vector<TensorPathStep> steps;
  steps.insert(steps.end(), a.getSteps().begin(), a.getSteps().end());
  steps.insert(steps.end(), b.getSteps().begin(), b.getSteps().end());
  Expr expr = op(a.getExpr(), b.getExpr());
  return MergeLatticePoint(steps, expr);
}

std::ostream& operator<<(std::ostream& os, const MergeLatticePoint& mlp) {
  vector<string> pathNames;
  if (mlp.getSteps().size() > 1) {
    os << "(";
  }
  os << util::join(mlp.getSteps(), " \u2227 ");
  if (mlp.getSteps().size() > 1) {
    os << ")";
  }
  os << "    " << mlp.getExpr() << std::endl;
  return os;
}

bool operator==(const MergeLatticePoint& a, const MergeLatticePoint& b) {
  auto& asteps = a.getSteps();
  auto& bsteps = b.getSteps();
  if (asteps.size() != bsteps.size()) {
    return false;
  }
  for (size_t i = 0; i < asteps.size(); i++) {
    if (asteps[i] != bsteps[i]) {
      return false;
    }
  }
  return true;
}

bool operator!=(const MergeLatticePoint& a, const MergeLatticePoint& b) {
  return !(a == b);
}

}}
