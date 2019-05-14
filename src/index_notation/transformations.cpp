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

IndexStmt TopoReorder::apply(IndexStmt stmt, std::string* reason) const {
  INIT_REASON(reason);

  // Build DAG of IndexVar dependencies (varDeps)
  struct DAGBuilder : public IndexNotationVisitor {
    using IndexNotationVisitor::visit;
    map<string, set<IndexVar>> tensorDeps;
    vector<pair<IndexVar, set<IndexVar>>> varDeps; // keep as vector so that on ties we pick original order
    map<Iterator, IndexVar> indexVars;
    IndexStmt innerBody;
    map <IndexVar, set<Forall::TAG>> forallTags;

    void visit(const ForallNode* node) {
      Forall foralli(node);
      IndexVar i = foralli.getIndexVar();
      Iterators iterators = Iterators::make(foralli, &indexVars);
      MergeLattice lattice = MergeLattice::make(foralli, iterators);

      forallTags[i] = foralli.getTags();

      auto findResult = std::find_if(varDeps.begin(), varDeps.end(), [i](const std::pair<IndexVar, set<IndexVar>>& e) {return e.first == i;});
      if (findResult == varDeps.end()) { // not found
        varDeps.push_back({i, {}});
      }

      // Add to varDeps and tensorDeps
      vector<Iterator> depIterators = lattice.iterators(); // ignore locaters
      depIterators.insert(depIterators.end(), lattice.results().begin(), lattice.results().end());

      for (Iterator iterator : depIterators) {
        if(iterator.getTensor().defined()) { // otherwise is dimension iterator
          IndexVar var = iterator.getIndexVar();
          string tensor = to<ir::Var>(iterator.getTensor())->name;
          if (tensorDeps.count(tensor)) {
            // Add any IndexVar dependencies
            auto varDep = std::find_if(varDeps.begin(), varDeps.end(), [var](const std::pair<IndexVar, set<IndexVar>>& e) {return e.first == var;});
            if (varDep == varDeps.end()) {
              // not found
              varDeps.push_back({var, set<IndexVar>(tensorDeps[tensor])});
            }
            else {
              varDep->second.insert(tensorDeps[tensor].begin(), tensorDeps[tensor].end());
            }
            cout << "New dep: " << *tensorDeps[tensor].begin() << " -> " << var << endl;
            // Add to tensorDeps
            tensorDeps[tensor].insert(i);
          }
          else {
            // Create new entry in tensorDeps
            tensorDeps[tensor] = {i};
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
  vector<pair<IndexVar, set<IndexVar>>> varDeps = dagBuilder.varDeps;

  // Topologicallly sort IndexVars
  vector<IndexVar> sortedVars;
  unsigned long countVars = varDeps.size();
  while (sortedVars.size() < countVars) {
    // Scan for variable with no dependencies
    IndexVar freeVar;
    size_t freeVarPos;
    for (freeVarPos = 0; freeVarPos < varDeps.size(); freeVarPos++) {
      pair<IndexVar, set<IndexVar>> varDep = varDeps[freeVarPos];
      IndexVar var = varDep.first;
      set<IndexVar> deps = varDep.second;
      if (deps.empty()) {
        freeVar = var;
        break;
      }
    }

    if (freeVarPos >= varDeps.size()) {
      // No free var found there is a cycle
      *reason = "Cycle exists in expression and a transpose is necessary. TACO does not yet support this"
                " you must manually transpose one or more of the tensors using the .transpose(...) method.";
      return IndexStmt();
    }
    cout << freeVar << ", ";
    sortedVars.push_back(freeVar);

    // remove dependencies on variable
    for (pair<IndexVar, set<IndexVar>> &varDep : varDeps) {
      set<IndexVar> &deps = varDep.second;
      if (deps.count(freeVar)) {
        deps.erase(freeVar);
        break;
      }
    }

    varDeps.erase(varDeps.begin() + freeVarPos);
  }
  cout << "\n";
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
