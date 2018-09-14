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
MergeLattice::MergeLattice(vector<MergePoint> points) : points(points) {
}

MergeLattice MergeLattice::make(Forall forall,
                                const map<ModeAccess,Iterator>& iterators) {
  struct MakeMergeLattice : public IndexNotationVisitorStrict {
    IndexVar i;
    map<ModeAccess,Iterator> iterators;
    MergeLattice lattice = MergeLattice({});

    MergeLattice makeLattice(IndexStmt stmt) {
      stmt.accept(this);
      MergeLattice l = lattice;
      lattice = MergeLattice({});
      return l;
    }

    MergeLattice makeLattice(IndexExpr expr) {
      expr.accept(this);
      MergeLattice l = lattice;
      lattice = MergeLattice({});
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
      MergePoint point = MergePoint({iterator}, {}, {});
      lattice = MergeLattice({point});
    }

    void visit(const LiteralNode* node) {
      lattice = MergeLattice({});
    }

    void visit(const NegNode* node) {
      lattice = makeLattice(node->a);
    }

    void visit(const AddNode* node) {
      MergeLattice a = makeLattice(node->a);
      MergeLattice b = makeLattice(node->b);
      if (a.defined() && b.defined()) {
        lattice = unionLattices(a, b);
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
        lattice = unionLattices(a, b);
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
        lattice = intersectLattices(a, b);
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
        lattice = intersectLattices(a, b);
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
      Iterator result = getIterator(node->lhs);

      // Add result to each point in l (as appender or inserter)
      vector<MergePoint> points;
      for (auto& point : l.getPoints()) {
        vector<Iterator> appenders;
        vector<Iterator> inserters;
        if (result.hasAppend()) {
          appenders.push_back(result);
        }
        else if (result.hasInsert()) {
          inserters.push_back(result);
        }
        else {
          taco_ierror << "Result must support insert or append";
        }
        points.push_back(MergePoint(point.getIterators(), point.getLocators(),
                                    {result}));
      }
      lattice = MergeLattice(points);
    }

    void visit(const ForallNode* node) {
      lattice = makeLattice(node);
    }

    void visit(const WhereNode* node) {
      taco_not_supported_yet;
    }

    void visit(const MultiNode* node) {
      lattice = unionLattices(makeLattice(node->stmt1),
                             makeLattice(node->stmt2));
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

MergeLattice MergeLattice::subLattice(MergePoint lp) const {
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
  return MergeLattice(dominatedPoints);
}

const vector<MergePoint>& MergeLattice::getPoints() const {
  return points;
}

const vector<Iterator>& MergeLattice::getIterators() const {
  // The iterators merged by a lattice are those merged by the first point
  taco_iassert(points.size() > 0) << "No merge points in the merge lattice";
  return points[0].getIterators();
}

const std::vector<Iterator>& MergeLattice::getResults() const {
  taco_iassert(points.size() > 0) << "No merge points in the merge lattice";
  return points[0].getResults();
}

bool MergeLattice::isExact() const {
  // A lattice is full if any merge point iterates over only full iterators
  // or if each sparse iterator is uniquely iterated by some lattice point.
  set<Iterator> uniquelyMergedIterators;
  for (auto& point : this->getPoints()) {
    if (all(point.getIterators(), [](Iterator it) {return it.isFull();})) {
      return true;
    }
  }

  for (auto& point : this->getPoints()) {
    if (point.getIterators().size() == 1) {
      uniquelyMergedIterators.insert(point.getIterators()[0]);
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

MergeLattice intersectLattices(MergeLattice a, MergeLattice b) {
  vector<MergePoint> points;

  // Append all combinations of a and b merge points
  for (auto& aLatticePoint : a.getPoints()) {
    for (auto& bLatticePoint : b.getPoints()) {
      points.push_back(intersectPoints(aLatticePoint, bLatticePoint));
    }
  }

  return MergeLattice(points);
}

MergeLattice unionLattices(MergeLattice a, MergeLattice b) {
  vector<MergePoint> points;

  // Append all combinations of the merge points of a and b
  vector<MergePoint> allPoints;
  for (auto& aLatticePoint : a.getPoints()) {
    for (auto& bLatticePoint : b.getPoints()) {
      allPoints.push_back(unionPoints(aLatticePoint, bLatticePoint));
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

  taco_iassert(points.size()>0) << "All lattices must have at least one point";
  MergeLattice lattice(points);
  return lattice;
}

ostream& operator<<(ostream& os, const MergeLattice& ml) {
  return os << util::join(ml.getPoints(), " \u2228\n");
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


// class MergePoint
struct MergePoint::Content {
  std::vector<Iterator> iterators;
  std::vector<Iterator> locators;
  std::vector<Iterator> results;
};

MergePoint::MergePoint(const vector<Iterator>& iterators,
                       const vector<Iterator>& locators,
                       const vector<Iterator>& results)
    : content(new Content) {
  content->iterators = iterators;
  content->locators = locators;
  content->results = results;
}

const vector<Iterator>& MergePoint::getIterators() const {
  return content->iterators;
}

const std::vector<Iterator>& MergePoint::getLocators() const {
  return content->locators;
}

const std::vector<Iterator>& MergePoint::getResults() const {
  return content->results;
}

MergePoint intersectPoints(MergePoint a, MergePoint b) {
  vector<Iterator> iterators = combine(a.getIterators(), b.getIterators());
  vector<Iterator> locaters  = combine(a.getLocators(), b.getLocators());
  vector<Iterator> results = combine(a.getResults(), b.getResults());
  return MergePoint(iterators, locaters, results);
}

MergePoint unionPoints(MergePoint a, MergePoint b) {
  vector<Iterator> iterators = combine(a.getIterators(), b.getIterators());
  vector<Iterator> locaters  = combine(a.getLocators(), b.getLocators());
  vector<Iterator> results = combine(a.getResults(), b.getResults());
  return MergePoint(iterators, locaters, results);
}

ostream& operator<<(ostream& os, const MergePoint& mlp) {
  return os << "["
            << util::join(mlp.getIterators(), ", ") << " | "
            << util::join(mlp.getLocators(),  ", ") << " | "
            << util::join(mlp.getResults(),   ", ")
            << "]";
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
  if (!compare(a.getIterators(), b.getIterators())) return false;
  if (!compare(a.getLocators(), b.getLocators())) return false;
  if (!compare(a.getResults(), b.getResults())) return false;
  return true;
}

bool operator!=(const MergePoint& a, const MergePoint& b) {
  return !(a == b);
}


// Free functions
std::pair<std::vector<Iterator>, std::vector<Iterator>>
splitRangersAndMergers(const std::vector<Iterator>& iterators) {
  vector<Iterator> rangers;
  vector<Iterator> mergers;

  // TODO: optimize this
  rangers = iterators;
  mergers = iterators;

  return {rangers, mergers};
}

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
