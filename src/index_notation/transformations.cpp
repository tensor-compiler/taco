#include "taco/index_notation/transformations.h"

#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_rewriter.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/error/error_messages.h"
#include "taco/util/collections.h"
#include "taco/lower/iterator.h"
#include "taco/lower/merge_lattice.h"

#include <iostream>
#include <algorithm>
#include <limits>

using namespace std;

namespace taco {


// class Transformation
Transformation::Transformation(Reorder reorder)
    : transformation(new Reorder(reorder)) {
}


Transformation::Transformation(Precompute precompute)
    : transformation(new Precompute(precompute)) {
}


Transformation::Transformation(Parallelize parallelize)
        : transformation(new Parallelize(parallelize)) {
}


IndexStmt Transformation::apply(IndexStmt stmt, string* reason) const {
  return transformation->apply(stmt, reason);
}


std::ostream& operator<<(std::ostream& os, const Transformation& t) {
  t.transformation->print(os);
  return os;
}


// class Reorder
struct Reorder::Content {
  IndexVar i;
  IndexVar j;
};


Reorder::Reorder(IndexVar i, IndexVar j) : content(new Content) {
  content->i = i;
  content->j = j;
}


IndexVar Reorder::geti() const {
  return content->i;
}


IndexVar Reorder::getj() const {
  return content->j;
}


IndexStmt Reorder::apply(IndexStmt stmt, string* reason) const {
  INIT_REASON(reason);

  string r;
  if (!isConcreteNotation(stmt, &r)) {
    *reason = "The index statement is not valid concrete index notation: " + r;
    return IndexStmt();
  }

  struct ReorderRewriter : public IndexNotationRewriter {
    using IndexNotationRewriter::visit;

    Reorder transformation;
    string* reason;
    ReorderRewriter(Reorder transformation, string* reason)
        : transformation(transformation), reason(reason) {}

    IndexStmt reorder(IndexStmt stmt) {
      IndexStmt reordered = rewrite(stmt);

      // Precondition: Did not find directly nested i,j loops
      if (reordered == stmt) {
        *reason = "The foralls of index variables " +
                  util::toString(transformation.geti()) + " and " +
                  util::toString(transformation.getj()) +
                  " are not directly nested.";
        return IndexStmt();
      }
      return reordered;
    }

    void visit(const ForallNode* node) {
      Forall foralli(node);
      
      IndexVar i = transformation.geti();
      IndexVar j = transformation.getj();

      // Nested loops with assignment or associative compound assignment.
      if ((foralli.getIndexVar() == i || foralli.getIndexVar() == j) &&
          isa<Forall>(foralli.getStmt())) {
        if (foralli.getIndexVar() == j) {
          swap(i, j);
        }
        auto forallj = to<Forall>(foralli.getStmt());
        if (forallj.getIndexVar() == j) {
          stmt = forall(j, forall(i, forallj.getStmt()));
          return;
        }
      }
      IndexNotationRewriter::visit(node);
    }
  };
  return ReorderRewriter(*this, reason).reorder(stmt);
}


void Reorder::print(std::ostream& os) const {
  os << "reorder(" << geti() << ", " << getj() << ")";
}


std::ostream& operator<<(std::ostream& os, const Reorder& reorder) {
  reorder.print(os);
  return os;
}


// class Precompute
struct Precompute::Content {
  IndexExpr expr;
  IndexVar i;
  IndexVar iw;
  TensorVar workspace;
};


Precompute::Precompute() : content(nullptr) {
}


Precompute::Precompute(IndexExpr expr, IndexVar i, IndexVar iw,
                     TensorVar workspace) : content(new Content) {
  content->expr = expr;
  content->i = i;
  content->iw = iw;
  content->workspace = workspace;
}


IndexExpr Precompute::getExpr() const {
  return content->expr;
}


IndexVar Precompute::geti() const {
  return content->i;
}


IndexVar Precompute::getiw() const {
  return content->iw;
}


TensorVar Precompute::getWorkspace() const {
  return content->workspace;
}


static bool containsExpr(Assignment assignment, IndexExpr expr) {
   struct ContainsVisitor : public IndexNotationVisitor {
    using IndexNotationVisitor::visit;

    IndexExpr expr;
    bool contains = false;

    void visit(const UnaryExprNode* node) {
      if (equals(IndexExpr(node), expr)) {
        contains = true;
      }
      else {
        IndexNotationVisitor::visit(node);
      }
    }

    void visit(const BinaryExprNode* node) {
      if (equals(IndexExpr(node), expr)) {
        contains = true;
      }
      else {
        IndexNotationVisitor::visit(node);
      }
    }

    void visit(const ReductionNode* node) {
      taco_ierror << "Reduction node in concrete index notation.";
    }
  };

  ContainsVisitor visitor;
  visitor.expr = expr;
  visitor.visit(assignment);
  return visitor.contains;
}


static Assignment getAssignmentContainingExpr(IndexStmt stmt, IndexExpr expr) {
  Assignment assignment;
  match(stmt,
        function<void(const AssignmentNode*,Matcher*)>([&assignment, &expr](
            const AssignmentNode* node, Matcher* ctx) {
          if (containsExpr(node, expr)) {
            assignment = node;
          }
        })
  );
  return assignment;
}


IndexStmt Precompute::apply(IndexStmt stmt, std::string* reason) const {
  INIT_REASON(reason);

  // Precondition: The expr to precompute is not in `stmt`
  Assignment assignment = getAssignmentContainingExpr(stmt, getExpr());
  if (!assignment.defined()) {
    *reason = "The expression (" + util::toString(getExpr()) + ") " +
              "is not in " + util::toString(stmt);
    return IndexStmt();
  }

  struct PrecomputeRewriter : public IndexNotationRewriter {
    using IndexNotationRewriter::visit;

    Precompute precompute;

    void visit(const ForallNode* node) {
      Forall foralli(node);
      IndexVar i = precompute.geti();

      if (foralli.getIndexVar() == i) {
        IndexStmt s = foralli.getStmt();
        TensorVar ws = precompute.getWorkspace();
        IndexExpr e = precompute.getExpr();
        IndexVar iw = precompute.getiw();

        IndexStmt consumer = forall(i, replace(s, {{e, ws(i)}}));
        IndexStmt producer = forall(iw, ws(iw) = replace(e, {{i,iw}}));
        Where where(consumer, producer);

        stmt = where;
        return;
      }
      IndexNotationRewriter::visit(node);
    }

  };
  PrecomputeRewriter rewriter;
  rewriter.precompute = *this;
  return rewriter.rewrite(stmt);
}


void Precompute::print(std::ostream& os) const {
  os << "precompute(" << getExpr() << ", " << geti() << ", "
     << getiw() << ", " << getWorkspace() << ")";
}


bool Precompute::defined() const {
  return content != nullptr;
}


std::ostream& operator<<(std::ostream& os, const Precompute& precompute) {
  precompute.print(os);
  return os;
}


// class Parallelize
struct Parallelize::Content {
  IndexVar i;
};


Parallelize::Parallelize() : content(nullptr) {
}


Parallelize::Parallelize(IndexVar i) : content(new Content) {
  content->i = i;
}


IndexVar Parallelize::geti() const {
  return content->i;
}


IndexStmt Parallelize::apply(IndexStmt stmt, std::string* reason) const {
  INIT_REASON(reason);

  struct ParallelizeRewriter : public IndexNotationRewriter {
    using IndexNotationRewriter::visit;

    Parallelize parallelize;
    std::string reason = "";

    void visit(const ForallNode* node) {
      Forall foralli(node);
      IndexVar i = parallelize.geti();

      Iterators iterators(foralli);
      MergeLattice lattice = MergeLattice::make(foralli, iterators);
      // Precondition 3: No parallelization of variables under a reduction
      // variable (ie MergePoint has at least 1 result iterators)
      if (lattice.results().empty()) {
        reason = "Precondition failed: Free variables cannot be dominated by reduction variables in the iteration graph, "
                 "as this causes scatter behavior and we do not yet emit parallel synchronization constructs";
        return;
      }

      if (foralli.getIndexVar() == i) {
        // Precondition 1: No coiteration of node (ie Merge Lattice has only 1 iterator)
        if (lattice.iterators().size() != 1) {
          reason = "Precondition failed: The loop must not merge tensor dimensions, that is, it must be a for loop;";
          return;
        }

        // Precondition 2: Every result iterator must have insert capability
        for (Iterator iterator : lattice.results()) {
          while (true) {
            if (!iterator.hasInsert()) {
              reason = "Precondition failed: The output tensor must allow inserts";
              return;
            }
            if (iterator.isLeaf()) {
              break;
            }
            iterator = iterator.getChild();
          }
        }

        stmt = forall(i, foralli.getStmt(), {Forall::PARALLELIZE});
        return;
      }
      IndexNotationRewriter::visit(node);
    }

  };

  ParallelizeRewriter rewriter;
  rewriter.parallelize = *this;
  IndexStmt rewritten = rewriter.rewrite(stmt);
  if (!rewriter.reason.empty()) {
    *reason = rewriter.reason;
    return IndexStmt();
  }
  return rewritten;
}


void Parallelize::print(std::ostream& os) const {
  os << "parallelize(" << geti() << ")";
}


std::ostream& operator<<(std::ostream& os, const Parallelize& parallelize) {
  parallelize.print(os);
  return os;
}


// Autoscheduling functions

IndexStmt parallelizeOuterLoop(IndexStmt stmt) {
  // get outer ForAll
  Forall forall;
  bool matched = false;
  match(stmt,
        function<void(const ForallNode*,Matcher*)>([&forall, &matched](
                const ForallNode* node, Matcher* ctx) {
          if (!matched) forall = node;
          matched = true;
        })
  );

  if (!matched) return stmt;
  string reason;
  IndexStmt parallelized = Parallelize(forall.getIndexVar()).apply(stmt, &reason);
  if (parallelized == IndexStmt()) {
    // can't parallelize
    return stmt;
  }
  return parallelized;
}

// Takes in a set of pairs of IndexVar and level for a given tensor and orders
// the IndexVars by tensor level
static vector<pair<IndexVar, bool>> 
varOrderFromTensorLevels(set<pair<IndexVar, pair<int, bool>>> tensorLevelVars) {
  vector<pair<IndexVar, pair<int, bool>>> sortedPairs(tensorLevelVars.begin(), 
                                                      tensorLevelVars.end());
  auto comparator = [](const pair<IndexVar, pair<int, bool>> &left, 
                       const pair<IndexVar, pair<int, bool>> &right) {
    return left.second.first < right.second.first;
  };
  std::sort(sortedPairs.begin(), sortedPairs.end(), comparator);

  vector<pair<IndexVar, bool>> varOrder;
  std::transform(sortedPairs.begin(),
                sortedPairs.end(),
                std::back_inserter(varOrder),
                [](const std::pair<IndexVar, pair<int, bool>>& p) {
                  return pair<IndexVar, bool>(p.first, p.second.second);
                });
  return varOrder;
}


// Takes in varOrders from many tensors and creates a map of dependencies between IndexVars
static map<IndexVar, set<IndexVar>>
depsFromVarOrders(map<string, vector<pair<IndexVar,bool>>> varOrders) {
  map<IndexVar, set<IndexVar>> deps;
  for (const auto& varOrderPair : varOrders) {
    const auto& varOrder = varOrderPair.second;
    for (auto firstit = varOrder.begin(); firstit != varOrder.end(); ++firstit) {
      for (auto secondit = firstit + 1; secondit != varOrder.end(); ++secondit) {
        if (firstit->second || secondit->second) { // one of the dimensions must enforce constraints
          if (deps.count(secondit->first)) {
            deps[secondit->first].insert(firstit->first);
          } else {
            deps[secondit->first] = {firstit->first};
          }
        }
      }
    }
  }
  return deps;
}


static vector<IndexVar>
topologicallySort(map<IndexVar,set<IndexVar>> hardDeps,
                  map<IndexVar,multiset<IndexVar>> softDeps,
                  vector<IndexVar> originalOrder) {
  vector<IndexVar> sortedVars;
  unsigned long countVars = originalOrder.size();
  while (sortedVars.size() < countVars) {
    // Scan for variable with no dependencies
    IndexVar freeVar;
    size_t freeVarPos = std::numeric_limits<size_t>::max();
    size_t minSoftDepsRemaining = std::numeric_limits<size_t>::max();
    for (size_t i = 0; i < originalOrder.size(); ++i) {
      IndexVar var = originalOrder[i];
      if (!hardDeps.count(var) || hardDeps[var].empty()) {  
        const size_t softDepsRemaining = softDeps.count(var) ? 
                                         softDeps[var].size() : 0;
        if (softDepsRemaining < minSoftDepsRemaining) {
          freeVar = var;
          freeVarPos = i;
          minSoftDepsRemaining = softDepsRemaining;
        }
      }
    }

    // No free var found there is a cycle
    taco_iassert(freeVarPos != std::numeric_limits<size_t>::max())
        << "Cycles in iteration graphs must be resolved, through transpose, "
        << "before the expression is passed to the topological sorting "
        << "routine.";

    sortedVars.push_back(freeVar);

    // remove dependencies on variable
    for (auto& varTensorDepsPair : hardDeps) {
      auto& varTensorDeps = varTensorDepsPair.second;
      if (varTensorDeps.count(freeVar)) {
        varTensorDeps.erase(freeVar);
      }
    }
    for (auto& varTensorDepsPair : softDeps) {
      auto& varTensorDeps = varTensorDepsPair.second;
      if (varTensorDeps.count(freeVar)) {
        varTensorDeps.erase(freeVar);
      }
    }
    originalOrder.erase(originalOrder.begin() + freeVarPos);
  }
  return sortedVars;
}


IndexStmt reorderLoopsTopologically(IndexStmt stmt) {
  // Collect tensorLevelVars which stores the pairs of IndexVar and tensor
  // level that each tensor is accessed at
  struct DAGBuilder : public IndexNotationVisitor {
    using IndexNotationVisitor::visit;
    // int is level, bool is if level enforces constraints (ie not dense)
    map<string, set<pair<IndexVar, pair<int, bool>>>> tensorLevelVars;
    IndexStmt innerBody;
    map <IndexVar, set<Forall::TAG>> forallTags;
    vector<IndexVar> indexVarOriginalOrder;
    Iterators iterators;

    DAGBuilder(Iterators iterators) : iterators(iterators) {};

    void visit(const ForallNode* node) {
      Forall foralli(node);
      IndexVar i = foralli.getIndexVar();

      MergeLattice lattice = MergeLattice::make(foralli, iterators);
      indexVarOriginalOrder.push_back(i);
      forallTags[i] = foralli.getTags();

      // Iterator and if Iterator enforces constraints
      vector<pair<Iterator, bool>> depIterators;
      for (Iterator iterator : lattice.points()[0].iterators()) {
        if (!iterator.isDimensionIterator()) {
          depIterators.push_back({iterator, true});
        }
      }

      for (Iterator iterator : lattice.points()[0].locators()) {
        depIterators.push_back({iterator, false});
      }

      // add result iterators that append
      for (Iterator iterator : lattice.results()) {
        depIterators.push_back({iterator, !iterator.hasInsert()});
      }

      for (const auto& iteratorPair : depIterators) {
        int level = iteratorPair.first.getMode().getLevel();
        string tensor = to<ir::Var>(iteratorPair.first.getTensor())->name;
        if (tensorLevelVars.count(tensor)) {
          tensorLevelVars[tensor].insert({{i, {level, iteratorPair.second}}});
        }
        else {
          tensorLevelVars[tensor] = {{{i, {level, iteratorPair.second}}}};
        }
      }

      if (!isa<Forall>(foralli.getStmt())) {
        innerBody = foralli.getStmt();
        return; // Only reorder first contiguous section of ForAlls
      }
      IndexNotationVisitor::visit(node);
    }
  };

  Iterators iterators(stmt);
  DAGBuilder dagBuilder(iterators);
  stmt.accept(&dagBuilder);

  // Construct tensor dependencies (sorted list of IndexVars) from tensorLevelVars
  map<string, vector<pair<IndexVar, bool>>> tensorVarOrders;
  for (const auto& tensorLevelVar : dagBuilder.tensorLevelVars) {
    tensorVarOrders[tensorLevelVar.first] = 
        varOrderFromTensorLevels(tensorLevelVar.second);
  }
  const auto hardDeps = depsFromVarOrders(tensorVarOrders);

  struct CollectSoftDependencies : public IndexNotationVisitor {
    using IndexNotationVisitor::visit;

    map<IndexVar, multiset<IndexVar>> softDeps;

    void visit(const AssignmentNode* op) {
      op->lhs.accept(this);
      op->rhs.accept(this);
    }

    void visit(const AccessNode* node) {
      const auto& modeOrdering = node->tensorVar.getFormat().getModeOrdering();
      for (size_t i = 1; i < (size_t)node->tensorVar.getOrder(); ++i) {
        const auto srcVar = node->indexVars[modeOrdering[i - 1]];
        const auto dstVar = node->indexVars[modeOrdering[i]];
        softDeps[dstVar].insert(srcVar);
      }
    }
  };
  CollectSoftDependencies collectSoftDeps;
  stmt.accept(&collectSoftDeps);

  const auto sortedVars = topologicallySort(hardDeps, collectSoftDeps.softDeps, 
                                            dagBuilder.indexVarOriginalOrder);

  // Reorder Foralls use a rewriter in case new nodes introduced outside of Forall
  struct TopoReorderRewriter : public IndexNotationRewriter {
    using IndexNotationRewriter::visit;

    const vector<IndexVar>& sortedVars;
    IndexStmt innerBody;
    const map<IndexVar, set<Forall::TAG>>& forallTags;

    TopoReorderRewriter(const vector<IndexVar>& sortedVars, IndexStmt innerBody,
                        const map<IndexVar, set<Forall::TAG>>& forallTags)
        : sortedVars(sortedVars), innerBody(innerBody), forallTags(forallTags) {
    }

    void visit(const ForallNode* node) {
      Forall foralli(node);
      IndexVar i = foralli.getIndexVar();

      // first forall must be in collected variables
      taco_iassert(util::contains(sortedVars, i));
      stmt = innerBody;
      for (auto it = sortedVars.rbegin(); it != sortedVars.rend(); ++it) {
        stmt = forall(*it, stmt, forallTags.at(*it));
      }
      return;
    }

  };
  TopoReorderRewriter rewriter(sortedVars, dagBuilder.innerBody, 
                               dagBuilder.forallTags);
  return rewriter.rewrite(stmt);
}

static bool compare(std::vector<IndexVar> vars1, std::vector<IndexVar> vars2) {
  return vars1 == vars2;
}

// TODO Temporary function to insert workspaces into SpMM kernels
static IndexStmt optimizeSpMM(IndexStmt stmt) {
  if (!isa<Forall>(stmt)) {
    return stmt;
  }
  Forall foralli = to<Forall>(stmt);
  IndexVar i = foralli.getIndexVar();

  if (!isa<Forall>(foralli.getStmt())) {
    return stmt;
  }
  Forall forallk = to<Forall>(foralli.getStmt());
  IndexVar k = forallk.getIndexVar();

  if (!isa<Forall>(forallk.getStmt())) {
    return stmt;
  }
  Forall forallj = to<Forall>(forallk.getStmt());
  IndexVar j = forallj.getIndexVar();

  if (!isa<Assignment>(forallj.getStmt())) {
    return stmt;
  }
  Assignment assignment = to<Assignment>(forallj.getStmt());

  if (!isa<Mul>(assignment.getRhs())) {
    return stmt;
  }
  Mul mul = to<Mul>(assignment.getRhs());

  taco_iassert(isa<Access>(assignment.getLhs()));
  if (!isa<Access>(mul.getA())) {
    return stmt;
  }
  if (!isa<Access>(mul.getB())) {
    return stmt;
  }

  Access Aaccess = to<Access>(assignment.getLhs());
  Access Baccess = to<Access>(mul.getA());
  Access Caccess = to<Access>(mul.getB());

  if (Aaccess.getIndexVars().size() != 2 ||
      Baccess.getIndexVars().size() != 2 ||
      Caccess.getIndexVars().size() != 2) {
    return stmt;
  }

  if (!compare(Aaccess.getIndexVars(), {i,j}) ||
      !compare(Baccess.getIndexVars(), {i,k}) ||
      !compare(Caccess.getIndexVars(), {k,j})) {
    return stmt;
  }

  TensorVar A = Aaccess.getTensorVar();
  if (A.getFormat().getModeFormats()[0].getName() != "dense" ||
      A.getFormat().getModeFormats()[1].getName() != "compressed" ||
      A.getFormat().getModeOrdering()[0] != 0 ||
      A.getFormat().getModeOrdering()[1] != 1) {
    return stmt;
  }

  TensorVar B = Baccess.getTensorVar();
  if (B.getFormat().getModeFormats()[0].getName() != "dense" ||
      B.getFormat().getModeFormats()[1].getName() != "compressed" ||
      B.getFormat().getModeOrdering()[0] != 0 ||
      B.getFormat().getModeOrdering()[1] != 1) {
    return stmt;
  }

  TensorVar C = Caccess.getTensorVar();
  if (C.getFormat().getModeFormats()[0].getName() != "dense" ||
      C.getFormat().getModeFormats()[1].getName() != "compressed" ||
      C.getFormat().getModeOrdering()[0] != 0 ||
      C.getFormat().getModeOrdering()[1] != 1) {
    return stmt;
  }

  // It's an SpMM statement so return an optimized SpMM statement
  TensorVar w("w",
              Type(Float64, {A.getType().getShape().getDimension(1)}),
              taco::dense);
  return forall(i,
                where(forall(j,
                             A(i,j) = w(j)),
                      forall(k,
                             forall(j,
                                    w(j) += B(i,k) * C(k,j)))));
}

IndexStmt insertTemporaries(IndexStmt stmt)
{
  IndexStmt spmm = optimizeSpMM(stmt);
  if (spmm != stmt) {
    return spmm;
  }

  // TODO Implement general workspacing when scattering into sparse result modes

  // Result dimensions that are indexed by free variables dominated by a
  // reduction variable are scattered into.  If any of these are compressed
  // then we introduce a dense workspace to scatter into instead.  The where
  // statement must push the reduction loop into the producer side, leaving
  // only the free variable loops on the consumer side.

  //vector<IndexVar> reductionVars = getReductionVars(stmt);
  //...

  return stmt;
}

}
