#include "taco/lower/merge_lattice.h"

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

// class MergeLattice
MergeLattice::MergeLattice() {
}

MergeLattice::MergeLattice(vector<MergePoint> points,
                           vector<Iterator> resultIterators)
    : mergePoint(points), resultIterators(resultIterators) {
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
      size_t loc = util::locate(access.getIndexVars(),i) + 1;
      taco_iassert(util::contains(iterators, ModeAccess(access,loc)))
          << "Cannot find " << ModeAccess(access,loc);
      return iterators.at(ModeAccess(access,loc));
    }

    void visit(const AccessNode* access) {
      Iterator iterator = getIterator(access);
      MergePoint latticePoint =
          MergePoint({iterator}, {iterator}, {iterator});
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
        lattice = latticeUnion(a, b);
      }
      // Scalar operands
      else if (a.defined()) {
        lattice = a;
      }
      else if (b.defined()) {
        lattice = b;
      }
    }

    void visit(const SubNode* expr) {
      MergeLattice a = makeLattice(expr->a);
      MergeLattice b = makeLattice(expr->b);
      if (a.defined() && b.defined()) {
        lattice = latticeUnion(a, b);
      }
      // Scalar operands
      else if (a.defined()) {
        lattice = a;
      }
      else if (b.defined()) {
        lattice = b;
      }
    }

    void visit(const MulNode* expr) {
      MergeLattice a = makeLattice(expr->a);
      MergeLattice b = makeLattice(expr->b);
      if (a.defined() && b.defined()) {
        lattice = latticeIntersection(a, b);
      }
      // Scalar operands
      else if (a.defined()) {
        lattice = a;
      }
      else if (b.defined()) {
        lattice = b;
      }
    }

    void visit(const DivNode* expr) {
      MergeLattice a = makeLattice(expr->a);
      MergeLattice b = makeLattice(expr->b);
      if (a.defined() && b.defined()) {
        lattice = latticeIntersection(a, b);
      }
      // Scalar operands
      else if (a.defined()) {
        lattice = a;
      }
      else if (b.defined()) {
        lattice = b;
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

MergeLattice latticeIntersection(MergeLattice a, MergeLattice b) {
  vector<MergePoint> points;

  // Append all combinations of a and b merge points
  for (auto& aLatticePoint : a.getPoints()) {
    for (auto& bLatticePoint : b.getPoints()) {
      points.push_back(pointIntersection(aLatticePoint, bLatticePoint));
    }
  }

  return MergeLattice(points, util::combine(a.getResultIterators(),
                                            b.getResultIterators()));
}

MergeLattice latticeUnion(MergeLattice a, MergeLattice b) {
  vector<MergePoint> points;

  // Append all combinations of the merge points of a and b
  vector<MergePoint> allPoints;
  for (auto& aLatticePoint : a.getPoints()) {
    for (auto& bLatticePoint : b.getPoints()) {
      allPoints.push_back(pointUnion(aLatticePoint, bLatticePoint));
    }
  }

  // Append the merge points of a
  util::append(allPoints, a.getPoints());

  // Append the merge points of b
  util::append(allPoints, b.getPoints());

  taco_iassert(allPoints.size()>0) << "A lattice must have at least one point";

  // Remove lattice points that can never be reached, as exhausting an iterator
  // over a full tensor mode cause the lattice to drop to zero.
  auto fullIterators = filter(allPoints[0].getIterators(), {ModeFormat::FULL});
  for (auto& point : allPoints) {
    bool missingFullIterator = false;
    for (auto& fullIterator : fullIterators) {
      if (!util::contains(point.getIterators(), fullIterator)) {
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
  return os << util::join(ml.getPoints(), " \u2228 ");
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
                       vector<Iterator> mergers)
    : iterators(iterators), mergers(mergers), rangers(rangers) {
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

static vector<Iterator> mergeRangers(vector<Iterator> a, vector<Iterator> b) {
  vector<Iterator> rangers = combine(a, b);

  // If only full iterators then return one of them, otherwise remove all the
  // full iterators.
  return all(rangers, [](Iterator iterator) {return iterator.isFull();})
         ? vector<Iterator>({rangers[0]})
         : filter(rangers, [](Iterator iterator) {return !iterator.isFull();});
}

static vector<Iterator> intersectMergers(vector<Iterator> a,
                                         vector<Iterator> b) {
  vector<Iterator> mergers = combine(a, b);

  if (all(mergers, [](Iterator i) {return i.isFull();})) {
    return vector<Iterator>({mergers[0]});
  }
  mergers = filter(mergers, [](Iterator i) {return !i.isFull();});
  return all(mergers, [](Iterator i) {return i.hasLocate();})
         ? vector<Iterator>({mergers[0]})
         : filter(mergers, [](Iterator i) {return !i.hasLocate();});
}

static vector<Iterator> unionMergers(vector<Iterator> a,
                                     vector<Iterator> b) {
  vector<Iterator> mergers = combine(a, b);
  return all(mergers, [](Iterator iterator) {return iterator.isFull();})
         ? vector<Iterator>({mergers[0]})
         : filter(mergers, [](Iterator iterator) {return !iterator.isFull();});
}

MergePoint pointIntersection(MergePoint a, MergePoint b) {
  vector<Iterator> iterators = combine(a.getIterators(), b.getIterators());
  vector<Iterator> mergers = intersectMergers(a.getMergers(), b.getMergers());
  vector<Iterator> rangers = mergeRangers(a.getRangers(), b.getRangers());
  return MergePoint(iterators, rangers, mergers);
}

MergePoint pointUnion(MergePoint a, MergePoint b) {
  vector<Iterator> iterators = combine(a.getIterators(), b.getIterators());
  vector<Iterator> rangers = mergeRangers(a.getRangers(), b.getRangers());
  vector<Iterator> mergers = unionMergers(a.getMergers(), b.getMergers());
  return MergePoint(iterators, rangers, mergers);
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

}
