#include "taco/index_notation/transformations.h"

#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_rewriter.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/error/error_messages.h"

#include <iostream>
#include <taco/lower/iterator.h>
#include <taco/lower/merge_lattice.h>

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

}
