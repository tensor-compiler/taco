#include "merge_lattice.h"

#include <algorithm>

#include "expr_nodes.h"
#include "expr_visitor.h"
#include "operator.h"

#include "internal_tensor.h"
#include "iteration_schedule.h"
#include "merge_rule.h"
#include "tensor_path.h"
#include "util/collections.h"
#include "util/strings.h"

using namespace std;

namespace taco {
namespace lower {

// class MergeLattice
MergeLattice::MergeLattice() {
}


MergeLattice::MergeLattice(MergeLatticePoint point)
    : MergeLattice(vector<MergeLatticePoint>({point})) {
}

MergeLattice::MergeLattice(vector<MergeLatticePoint> points)
    : points(points) {
}

// TODO: Build lattice directly from Expr?
MergeLattice MergeLattice::make(const MergeRule& rule) {
  struct MergeLatticeVisitor : public MergeRuleVisitor {

    MergeLattice mergeLattice;
    MergeLattice buildMergeLattice(const MergeRule& rule) {
      rule.accept(this);
      return mergeLattice;
    }

    void visit(const Step* rule) {
      mergeLattice = MergeLatticePoint(rule->step, rule->expr);
    }

    void visit(const And* rule) {
      MergeLattice a = buildMergeLattice(rule->a);
      MergeLattice b = buildMergeLattice(rule->b);
      struct ConjunctionVisitor : internal::ExprVisitor {
        using internal::ExprVisitor::visit;

        MergeLattice a;
        MergeLattice b;
        ConjunctionVisitor(MergeLattice a, MergeLattice b) : a(a), b(b) {}

        MergeLattice lattice;
        MergeLattice getLattice(Expr expr) {
          expr.accept(this);
          return lattice;
        }

        void visit(const internal::Mul*) {
          lattice = conjunction<Mul>(a, b);
        }

        void visit(const internal::Div*) {
          lattice = conjunction<Div>(a, b);
        }
      };
      ConjunctionVisitor visitor(a, b);
      mergeLattice = visitor.getLattice(rule->expr);
    }

    void visit(const Or* rule) {
      MergeLattice a = buildMergeLattice(rule->a);
      MergeLattice b = buildMergeLattice(rule->b);
      struct DisjunctionVisitor : internal::ExprVisitor {
        using internal::ExprVisitor::visit;

        MergeLattice a;
        MergeLattice b;
        DisjunctionVisitor(MergeLattice a, MergeLattice b) : a(a), b(b) {}

        MergeLattice lattice;
        MergeLattice getLattice(Expr expr) {
          expr.accept(this);
          return lattice;
        }

        void visit(const internal::Add*) {
          lattice = disjunction<Add>(a, b);
        }

        void visit(const internal::Sub*) {
          lattice = disjunction<Sub>(a, b);
        }
      };
      DisjunctionVisitor visitor(a, b);
      mergeLattice = visitor.getLattice(rule->expr);
    }
  };
  MergeLattice mergeLattice = MergeLatticeVisitor().buildMergeLattice(rule);
  return mergeLattice;
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


// class MergeLatticePoint
MergeLatticePoint::MergeLatticePoint(const TensorPathStep& step,
                                     const Expr& expr)
    : MergeLatticePoint(vector<TensorPathStep>({step}), expr) {
}

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
  return os;
}

}}
