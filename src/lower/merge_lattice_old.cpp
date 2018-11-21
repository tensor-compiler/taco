#include "merge_lattice_old.h"

#include <set>
#include <algorithm>

#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/index_notation/index_notation_visitor.h"
#include "tensor_path.h"
#include "iteration_graph.h"
#include "iterators.h"
#include "mode_access.h"
#include "taco/util/collections.h"
#include "taco/util/strings.h"

using namespace std;

namespace taco {
namespace old {

// class MergeLattice
MergeLattice::MergeLattice() {
}

MergeLattice::MergeLattice(vector<MergePoint> points,
                           vector<Iterator> resultIterators)
    : mergePoint(points), resultIterators(resultIterators) {
}

template <class op>
static
MergeLattice scale(MergeLattice lattice, IndexExpr scale, bool leftScale) {
  vector<MergePoint> scaledPoints;
  for (auto& point : lattice.getPoints()) {
    IndexExpr expr = point.getExpr();
    IndexExpr scaledExpr = (leftScale) ? new op(scale, expr)
                                       : new op(expr, scale);
    MergePoint scaledPoint(point.getIterators(), point.getRangers(),
                           point.getMergers(), scaledExpr);
    scaledPoints.push_back(scaledPoint);
  }
  return MergeLattice(scaledPoints, lattice.getRangeIterators());
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
  vector<MergePoint> negPoints;
  for (auto& point : lattice.getPoints()) {
    IndexExpr negExpr = new op(point.getExpr());
    negPoints.push_back(MergePoint(point.getIterators(), point.getRangers(),
                                   point.getMergers(), negExpr));
  }
  return MergeLattice(negPoints, lattice.getRangeIterators());
}

MergeLattice MergeLattice::make(Forall forall,
                                const map<ModeAccess,Iterator>& iterators) {
  struct MakeMergeLattice : public IndexNotationVisitorStrict {
    IndexVar i;
    map<ModeAccess,Iterator> iterators;
    MergeLattice lattice;

    MergeLattice makeLattice(IndexStmt stmt) {
      stmt.accept(this);
      MergeLattice l = lattice;
      lattice = MergeLattice();
      return l;
    }

    MergeLattice makeLattice(IndexExpr expr) {
      expr.accept(this);
      MergeLattice l = lattice;
      lattice = MergeLattice();
      return l;
    }

    Iterator getIterator(Access access) {
      int loc = (int)util::locate(access.getIndexVars(),i) + 1;
      taco_iassert(util::contains(iterators, ModeAccess(access,loc)));
      return iterators.at(ModeAccess(access,loc));
    }

    void visit(const AccessNode* access) {
      Iterator iterator = getIterator(access);
      MergePoint latticePoint =
          MergePoint({iterator}, {iterator}, {iterator}, access);
      lattice = MergeLattice({latticePoint}, {});
    }

    void visit(const LiteralNode* node) {
      taco_not_supported_yet;
    }

    void visit(const NegNode* node) {
      lattice = makeLattice(node->a);
    }

    void visit(const AddNode* node) {
      MergeLattice a = makeLattice(node->a);
      MergeLattice b = makeLattice(node->b);
      if (a.defined() && b.defined()) {
        lattice = mergeUnion<AddNode>(a, b);
      }
      // Scalar operands
      else if (a.defined()) {
        lattice = scale<AddNode>(a, node->b);
      }
      else if (b.defined()) {
        lattice = scale<AddNode>(node->a, b);
      }
    }

    void visit(const SubNode* expr) {
      MergeLattice a = makeLattice(expr->a);
      MergeLattice b = makeLattice(expr->b);
      if (a.defined() && b.defined()) {
        lattice = mergeUnion<SubNode>(a, b);
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
      MergeLattice a = makeLattice(expr->a);
      MergeLattice b = makeLattice(expr->b);
      if (a.defined() && b.defined()) {
        lattice = mergeIntersection<MulNode>(a, b);
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
      MergeLattice a = makeLattice(expr->a);
      MergeLattice b = makeLattice(expr->b);
      if (a.defined() && b.defined()) {
        lattice = mergeIntersection<DivNode>(a, b);
      }
      // Scalar operands
      else if (a.defined()) {
        lattice = scale<DivNode>(a, expr->b);
      }
      else if (b.defined()) {
        lattice = scale<DivNode>(expr->a, b);
      }
    }

    void visit(const SqrtNode* expr) {
      lattice = makeLattice(expr->a);
    }

    void visit(const ReductionNode* node) {
      taco_ierror << "Merge lattices must be created from concrete index "
                  << "notation, which does not have reduction nodes.";
    }

    void visit(const AssignmentNode* node) {
      MergeLattice l = makeLattice(node->rhs);
      lattice = MergeLattice(l.getPoints(), {getIterator(node->lhs)});
    }

    void visit(const ForallNode* node) {
      taco_not_supported_yet;
    }

    void visit(const WhereNode* node) {
      taco_not_supported_yet;
    }

    void visit(const MultiNode* node) {
      taco_not_supported_yet;
    }

    void visit(const SequenceNode* node) {
      taco_not_supported_yet;
    }
  };

  MakeMergeLattice make;
  make.i = forall.getIndexVar();
  make.iterators = iterators;
  return make.makeLattice(forall.getStmt());
}

MergeLattice MergeLattice::make(const IndexExpr& indexExpr,
                                const IndexVar& indexVar,
                                const old::IterationGraph& iterationGraph,
                                const old::Iterators& iterators) {

  struct BuildMergeLattice : public IndexExprVisitorStrict {
    const IndexVar&            indexVar;
    const old::IterationGraph& iterationGraph;
    const old::Iterators&      iterators;
    MergeLattice               lattice;

    BuildMergeLattice(const IndexVar& indexVar,
                      const old::IterationGraph& iterationGraph,
                      const old::Iterators& iterators)
        : indexVar(indexVar), iterationGraph(iterationGraph),
          iterators(iterators) {
    }

    MergeLattice buildLattice(const IndexExpr& expr) {
      expr.accept(this);
      MergeLattice l = lattice;
      lattice = MergeLattice();
      return l;
    }

    using IndexExprVisitorStrict::visit;

    void visit(const AccessNode* expr) {
      // Throw away expressions `indexVar` does not contribute to
      if (!util::contains(expr->indexVars, indexVar)) {
        lattice = MergeLattice();
        return;
      }

      old::TensorPath path = iterationGraph.getTensorPath(expr);
      size_t i = util::locate(path.getVariables(), indexVar);
      Iterator iterator = iterators[path.getStep(i)];
      auto latticePoint = MergePoint({iterator}, {iterator}, {iterator}, expr);
      lattice = MergeLattice({latticePoint}, {});
    }

    void visit(const LiteralNode*) {
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
        lattice = mergeUnion<AddNode>(a, b);
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
        lattice = mergeUnion<SubNode>(a, b);
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
        lattice = mergeIntersection<MulNode>(a, b);
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
        lattice = mergeIntersection<DivNode>(a, b);
      }
      // Scalar operands
      else if (a.defined()) {
        lattice = scale<DivNode>(a, expr->b);
      }
      else if (b.defined()) {
        lattice = scale<DivNode>(expr->a, b);
      }
    }

    void visit(const ReductionNode* expr) {
      lattice = buildLattice(expr->a);
    }
  };

  auto lattice = BuildMergeLattice(indexVar, iterationGraph,
                                   iterators).buildLattice(indexExpr);
  taco_iassert(lattice.getSize() > 0) <<
      "Every merge lattice should have at least one merge point";
  return lattice;
}

size_t MergeLattice::getSize() const {
  return mergePoint.size();
}

const MergePoint& MergeLattice::operator[](size_t i) const {
  return mergePoint[i];
}

const vector<MergePoint>& MergeLattice::getPoints() const {
  return mergePoint;
}

const vector<Iterator>& MergeLattice::getIterators() const {
  // The iterators merged by a lattice are those merged by the first point
  taco_iassert(mergePoint.size() > 0) << "No merge points in the merge lattice";
  return mergePoint[0].getIterators();
}

const vector<Iterator>& MergeLattice::getRangeIterators() const {
  // The iterators merged by a lattice are those merged by the first point
  taco_iassert(mergePoint.size() > 0) << "No merge points in the merge lattice";
  return mergePoint[0].getRangers();
}

const vector<Iterator>& MergeLattice::getResultIterators() const {
  return resultIterators;
}

const IndexExpr& MergeLattice::getExpr() const {
  taco_iassert(mergePoint.size() > 0) << "No merge points in the merge lattice";

  // The expression merged by a lattice is the same as the expression of the
  // first merge point
  return mergePoint[0].getExpr();
}

MergeLattice MergeLattice::getSubLattice(MergePoint lp) const {
  // A merge point lp dominats lq iff it contains a subset of lp's
  // tensor path steps. So we scan through the points and filter those points.
  vector<MergePoint> dominatedPoints;
  vector<Iterator> lpIterators = lp.getIterators();
  sort(lpIterators.begin(), lpIterators.end());
  for (auto& lq : this->getPoints()) {
    vector<Iterator> lqIterators = lq.getIterators();
    sort(lqIterators.begin(), lqIterators.end());
    if (includes(lpIterators.begin(), lpIterators.end(),
                      lqIterators.begin(), lqIterators.end())) {
      dominatedPoints.push_back(lq);
    }
  }
  return MergeLattice(dominatedPoints, resultIterators);
}

bool MergeLattice::isFull() const {
  // A merge lattice is full if any merge point iterates over a single full
  // iterator or if each sparse iterator is uniquely iterated by some lattice 
  // point.
  set<Iterator> uniquelyMergedIterators;
  for (auto& point : this->getPoints()) {
    if (point.getRangers().size() == 1) {
      auto it = point.getRangers()[0];
      uniquelyMergedIterators.insert(it);
      if (it.isFull()) {
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
  return mergePoint.size() > 0;
}

template<class op>
MergeLattice mergeIntersection(MergeLattice a, MergeLattice b) {
  vector<MergePoint> points;

  // Append all combinations of a and b merge points
  for (auto& aLatticePoint : a.getPoints()) {
    for (auto& bLatticePoint : b.getPoints()) {
      points.push_back(mergeIntersection<op>(aLatticePoint, bLatticePoint));
    }
  }

  return MergeLattice(points, util::combine(a.getResultIterators(),
                                            b.getResultIterators()));
}

template<class op>
MergeLattice mergeUnion(MergeLattice a, MergeLattice b) {
  vector<MergePoint> points;

  // Append all combinations of the merge points of a and b
  vector<MergePoint> allPoints;
  for (auto& aLatticePoint : a.getPoints()) {
    for (auto& bLatticePoint : b.getPoints()) {
      allPoints.push_back(mergeUnion<op>(aLatticePoint, bLatticePoint));
    }
  }

  // Append the merge points of a
  util::append(allPoints, a.getPoints());

  // Append the merge points of b
  for (auto &bLatticePoint : b.getPoints()) {
    Datatype type = bLatticePoint.getExpr().getDataType();
    IndexExpr expr = new op(Literal::zero(type), bLatticePoint.getExpr());
    allPoints.push_back(MergePoint(bLatticePoint.getIterators(),
                        bLatticePoint.getRangers(),
                        bLatticePoint.getMergers(),
                        expr));
  }

  taco_iassert(allPoints.size() > 0) <<
      "A lattice must have at least one point";

  // Exhausting an iterator over a full tensor mode cause the lattice to drop
  // to zero. Therefore we cannot end up in a merge point that doesn't
  // contain the iterator over the full mode and must remove all merge points
  // that don't contain it.
  auto fullIterators = old::getFullIterators(allPoints[0].getMergers());
  for (auto& point : allPoints) {
    bool missingFullIterator = false;
    for (auto& fullIterator : fullIterators) {
      if (!util::contains(point.getMergers(), fullIterator)) {
        missingFullIterator = true;
        break;
      }
    }
    if (!missingFullIterator) {
      points.push_back(point);
    }
  }

  MergeLattice lattice(points, util::combine(a.getResultIterators(),
                                             b.getResultIterators()));
  taco_iassert(lattice.getSize() > 0) <<
      "All lattices must have at least one point";
  return lattice;
}

ostream& operator<<(ostream& os, const MergeLattice& ml) {
  return os << util::join(ml.getPoints(), " \u2228\n");
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
MergePoint::MergePoint(vector<Iterator> iterators, vector<Iterator> rangers,
                       vector<Iterator> mergers, IndexExpr expr)
    : iterators(iterators), mergers(mergers),
      rangers(rangers), expr(expr) {
  taco_iassert(iterators.size() >= mergers.size());
  taco_iassert(mergers.size() >= rangers.size());
}

const vector<Iterator>& MergePoint::getIterators() const {
  return iterators;
}

const vector<Iterator>& MergePoint::getRangers() const {
  return rangers;
}

const vector<Iterator>& MergePoint::getMergers() const {
  return mergers;
}

const IndexExpr& MergePoint::getExpr() const {
  return expr;
}

template<class op>
MergePoint mergeIntersection(MergePoint a, MergePoint b) {
  vector<Iterator> iterators = combine(a.getIterators(), b.getIterators());

  IndexExpr expr = new op(a.getExpr(), b.getExpr());

  vector<Iterator> aMergers = a.getMergers();
  for (const auto& iter : b.getMergers()) {
    if (!iter.hasLocate()) {
      aMergers.push_back(iter);
    }
  }

  vector<Iterator> bMergers = b.getMergers();
  for (const auto& iter : a.getMergers()) {
    if (!iter.hasLocate()) {
      bMergers.push_back(iter);
    }
  }

  vector<Iterator> aRangers = simplify(aMergers);
  vector<Iterator> bRangers = simplify(bMergers);

  MergePoint point = (aRangers.size() <= bRangers.size())
          ? MergePoint(iterators, aRangers, aMergers, expr)
          : MergePoint(iterators, bRangers, bMergers, expr);

  return point;
}

template<class op>
MergePoint mergeUnion(MergePoint a, MergePoint b) {
  vector<Iterator> iterators = combine(a.getIterators(), b.getIterators());

  IndexExpr expr = new op(a.getExpr(), b.getExpr());

  vector<Iterator> aMergers = a.getMergers();
  for (const auto& iter : b.getMergers()) {
    aMergers.push_back(iter);
  }

  vector<Iterator> bMergers = b.getMergers();
  for (const auto& iter : a.getMergers()) {
    bMergers.push_back(iter);
  }

  vector<Iterator> aRangers = simplify(aMergers);
  vector<Iterator> bRangers = simplify(bMergers);

  MergePoint point = (aRangers.size() <= bRangers.size())
          ? MergePoint(iterators, aRangers, aMergers, expr)
          : MergePoint(iterators, bRangers, bMergers, expr);

  return point;
}

ostream& operator<<(ostream& os, const MergePoint& mlp) {
  return os << "["
            << util::join(mlp.getIterators(), " \u2227 ") << " | "
            << util::join(mlp.getRangers(),   " \u2227 ") << " | "
            << util::join(mlp.getMergers(),   " \u2227 ")
            << "]";
}

static bool compare(const vector<Iterator>& a, const vector<Iterator>& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (size_t i = 0; i < a.size(); i++) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

bool operator==(const MergePoint& a, const MergePoint& b) {
  if (!compare(a.getIterators(), b.getIterators())) return false;
  if (!compare(a.getRangers(), b.getRangers())) return false;
  if (!compare(a.getMergers(), b.getMergers())) return false;
  return true;
}

bool operator!=(const MergePoint& a, const MergePoint& b) {
  return !(a == b);
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

set<Access> exhaustedAccesses(MergePoint lp, MergeLattice l) {
  set<Iterator> notExhaustedIters(lp.getIterators().begin(),
                                       lp.getIterators().end());
  set<Access> exhausted;
  for (auto& iter : l.getIterators()) {
    if (!util::contains(notExhaustedIters, iter)) {
      exhausted.insert(iter.getTensorPath().getAccess());
    }
  }

  return exhausted;
}

}}
