#include "taco/index_notation/transformations.h"

#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_rewriter.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/error/error_messages.h"

#include <iostream>
#include <taco/lower/iterator.h>
#include <taco/lower/merge_lattice.h>
#include <algorithm>

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

Transformation::Transformation(TopoReorder topo_reorder)
        : transformation(new TopoReorder(topo_reorder)) {
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
    map<Iterator, IndexVar> indexVars;

    void visit(const ForallNode* node) {
      Forall foralli(node);
      IndexVar i = parallelize.geti();

      Iterators iterators = Iterators::make(foralli, &indexVars);
      MergeLattice lattice = MergeLattice::make(foralli, iterators);
      // Precondition 3: No parallelization of variables under a reduction variable (ie MergePoint has at least 1 result iterators)
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
          if (!iterator.hasInsert()) {
            reason = "Precondition failed: The output tensor must allow inserts";
            return;
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


TopoReorder::TopoReorder() {
}

// Takes in a set of pairs of IndexVar and level for a given tensor and orders the IndexVars by tensor level
static vector<IndexVar> varOrderFromTensorLevels(set<pair<IndexVar, int>> tensorLevelVars) {
  vector<pair<IndexVar, int>> sortedPairs(tensorLevelVars.begin(), tensorLevelVars.end());
  std::sort(sortedPairs.begin(), sortedPairs.end(), [](pair<IndexVar, int> &left, pair<IndexVar, int> &right) {
    return left.second < right.second;
  });

  vector<IndexVar> varOrder;
  std::transform(sortedPairs.begin(),
                sortedPairs.end(),
                std::back_inserter(varOrder),
                [](const std::pair<IndexVar, int>& p) { return p.first; });
  return varOrder;
}

// Takes in varOrders from many tensors and creates a map of dependencies between IndexVars
static map<IndexVar, set<IndexVar>> depsFromVarOrders(map<string, vector<IndexVar>> varOrders) {
  map<IndexVar, set<IndexVar>> deps;
  for (pair<string, vector<IndexVar>> varOrderPair : varOrders) {
    vector<IndexVar> varOrder = varOrderPair.second;
    for (auto firstit = varOrder.begin(); firstit != varOrder.end(); ++firstit) {
      for (auto secondit = firstit + 1; secondit != varOrder.end(); ++secondit) {
        cout << "New Dep: " << *firstit << " -> " << *secondit << endl;
        if (deps.count(*secondit)) {
          deps[*secondit].insert(*firstit);
        }
        else {
          deps[*secondit] = {*firstit};
        }
      }
    }
  }
  return deps;
}

static vector<IndexVar> topologicallySort(map<IndexVar, set<IndexVar>>  tensorDeps, vector<IndexVar> originalOrder, bool &cycle) {
  vector<IndexVar> sortedVars;
  unsigned long countVars = originalOrder.size();
  while (sortedVars.size() < countVars) {
    // Scan for variable with no dependencies
    IndexVar freeVar;
    size_t freeVarPos;
    for (freeVarPos = 0; freeVarPos < originalOrder.size(); freeVarPos++) {
      IndexVar var = originalOrder[freeVarPos];
      if (!tensorDeps.count(var) || tensorDeps[var].empty()) {
        freeVar = var;
        break;
      }
    }

    if (freeVarPos >= originalOrder.size()) {
      // No free var found there is a cycle
      cycle = true;
      return {};
    }
    cout << freeVar << ", ";
    sortedVars.push_back(freeVar);

    // remove dependencies on variable
    for (pair<const IndexVar, set<IndexVar>> &varTensorDepsPair : tensorDeps) {
      set<IndexVar> &varTensorDeps = varTensorDepsPair.second;
      if (varTensorDeps.count(freeVar)) {
        varTensorDeps.erase(freeVar);
      }
    }
    originalOrder.erase(originalOrder.begin() + freeVarPos);
  }
  cycle = false;
  return sortedVars;
}

IndexStmt TopoReorder::apply(IndexStmt stmt, std::string* reason) const {
  INIT_REASON(reason);

  // Collect tensorLevelVars which stores the pairs of IndexVar and tensor level that each tensor is accessed at
  struct DAGBuilder : public IndexNotationVisitor {
    using IndexNotationVisitor::visit;
    map<string, set<pair<IndexVar, int>>> tensorLevelVars;
    map<Iterator, IndexVar> indexVars;
    IndexStmt innerBody;
    map <IndexVar, set<Forall::TAG>> forallTags;
    vector<IndexVar> indexVarOriginalOrder;

    void visit(const ForallNode* node) {
      Forall foralli(node);
      IndexVar i = foralli.getIndexVar();
      Iterators iterators = Iterators::make(foralli, &indexVars);
      MergeLattice lattice = MergeLattice::make(foralli, iterators);

      indexVarOriginalOrder.push_back(i);
      forallTags[i] = foralli.getTags();

      vector<Iterator> depIterators = lattice.iterators(); // ignore locaters
      depIterators.insert(depIterators.end(), lattice.results().begin(), lattice.results().end());

      for (Iterator iterator : depIterators) {
        if(iterator.getTensor().defined()) { // otherwise is dimension iterator
          int level = iterator.getMode().getLevel();
          string tensor = to<ir::Var>(iterator.getTensor())->name;
          if (tensorLevelVars.count(tensor)) {
            tensorLevelVars[tensor].insert({{i, level}});
          }
          else {
            tensorLevelVars[tensor] = {{{i, level}}};
          }
        }
      }

      if (!isa<Forall>(foralli.getStmt())) {
        innerBody = foralli.getStmt();
        return; // Only reorder first contiguous section of ForAlls
      }
      IndexNotationVisitor::visit(node);
    }
  };

  DAGBuilder dagBuilder;
  stmt.accept(&dagBuilder);

  // Construct tensor dependencies (sorted list of IndexVars) from tensorLevelVars
  map<string, vector<IndexVar>> tensorVarOrders;
  for (pair<string, set<pair<IndexVar, int>>> tensorLevelVar : dagBuilder.tensorLevelVars) {
    tensorVarOrders[tensorLevelVar.first] = varOrderFromTensorLevels(tensorLevelVar.second);
  }

  map<IndexVar, set<IndexVar>> deps = depsFromVarOrders(tensorVarOrders);

  bool cycle;
  vector<IndexVar> sortedVars = topologicallySort(deps, dagBuilder.indexVarOriginalOrder, cycle);

  cout << "First sorted: " << sortedVars[0] << endl;

  if (cycle) {
    *reason = "Cycle exists in expression and a transpose is necessary. TACO does not yet support this"
              " you must manually transpose one or more of the tensors using the .transpose(...) method.";
    return IndexStmt();
  }

  // Reorder Foralls use a rewriter in case new nodes introduced outside of Forall
  struct TopoReorderRewriter : public IndexNotationRewriter {
    using IndexNotationRewriter::visit;

    vector<IndexVar> sortedVars;
    IndexStmt innerBody;
    map<IndexVar, set<Forall::TAG>> forallTags;

    TopoReorderRewriter(vector<IndexVar> sortedVars,
            IndexStmt innerBody,
            map<IndexVar, set<Forall::TAG>> forallTags)
            : sortedVars(sortedVars), innerBody(innerBody), forallTags(forallTags) {
    }

    void visit(const ForallNode* node) {
      Forall foralli(node);
      IndexVar i = foralli.getIndexVar();

      // first forall must be in collected variables
      taco_iassert(std::find(sortedVars.begin(), sortedVars.end(), i) != sortedVars.end());
      stmt = innerBody;
      for (auto it = sortedVars.rbegin(); it != sortedVars.rend(); ++it) {
        stmt = forall(*it, stmt, forallTags[*it]);
      }
      return;
    }

  };
  TopoReorderRewriter rewriter(sortedVars, dagBuilder.innerBody, dagBuilder.forallTags);
  return rewriter.rewrite(stmt);
}

void TopoReorder::print(std::ostream& os) const {
  os << "topo_reorder()";
}

std::ostream& operator<<(std::ostream& os, const TopoReorder& parallelize) {
  parallelize.print(os);
  return os;
}

}
