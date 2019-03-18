#include "taco/index_notation/transformations.h"

#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_rewriter.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/error/error_messages.h"

#include <iostream>

using namespace std;

namespace taco {

// class Transformation
Transformation::Transformation(Reorder reorder)
    : transformation(new Reorder(reorder)) {
}

Transformation::Transformation(Precompute precompute)
    : transformation(new Precompute(precompute)) {
}

Transformation::Transformation(ForAllReplace forallreplace)
        : transformation(new ForAllReplace(forallreplace)) {
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

// class ForAllReplace
struct ForAllReplace::Content {
  std::vector<IndexVar> pattern;
  std::vector<IndexVar> replacement;
};

ForAllReplace::ForAllReplace() : content(nullptr) {
}

ForAllReplace::ForAllReplace(std::vector<IndexVar> pattern, std::vector<IndexVar> replacement) : content(new Content) {
  taco_iassert(!pattern.empty());
  content->pattern = pattern;
  content->replacement = replacement;
}

std::vector<IndexVar> ForAllReplace::getPattern() const {
  return content->pattern;
}

std::vector<IndexVar> ForAllReplace::getReplacement() const {
  return content->replacement;
}

IndexStmt ForAllReplace::apply(IndexStmt stmt, string* reason) const {
  INIT_REASON(reason);

  string r;
  if (!isConcreteNotation(stmt, &r)) {
    *reason = "The index statement is not valid concrete index notation: " + r;
    return IndexStmt();
  }

  /// Since all IndexVars can only appear once, assume replacement will work and error if it doesn't
  struct ForAllReplaceRewriter : public IndexNotationRewriter {
    using IndexNotationRewriter::visit;

    ForAllReplace transformation;
    string* reason;
    int elementsMatched = 0;
    ForAllReplaceRewriter(ForAllReplace transformation, string* reason)
            : transformation(transformation), reason(reason) {}

    IndexStmt forallreplace(IndexStmt stmt) {
      IndexStmt replaced = rewrite(stmt);

      // Precondition: Did not find pattern
      if (replaced == stmt || elementsMatched == -1) {
        *reason = "The pattern of ForAlls: " +
                  util::join(transformation.getPattern()) +
                  " was not found while attempting to replace with: " +
                  util::join(transformation.getReplacement());
        return IndexStmt();
      }
      return replaced;
    }

    void visit(const ForallNode* node) {
      Forall foralli(node);
      vector<IndexVar> pattern = transformation.getPattern();
      if (elementsMatched == -1) {
        return; // pattern did not match
      }

      if(elementsMatched >= (int) pattern.size()) {
        IndexNotationRewriter::visit(node);
        return;
      }

      if (foralli.getIndexVar() == pattern[elementsMatched]) {
        // assume rest of pattern matches
        vector<IndexVar> replacement = transformation.getReplacement();
        bool firstMatch = (elementsMatched == 0);
        elementsMatched++;
        stmt = rewrite(foralli.getStmt());
        if (firstMatch) {
          // add replacement nodes and cut out this node
          for (auto i = replacement.rbegin(); i != replacement.rend(); ++i ) {
            stmt = forall(*i, stmt);
          }
        }
        // else cut out this node
        return;
      }
      else if (elementsMatched > 0) {
        elementsMatched = -1; // pattern did not match
        return;
      }
      // before pattern match
      IndexNotationRewriter::visit(node);
    }
  };
  return ForAllReplaceRewriter(*this, reason).forallreplace(stmt);
}

void ForAllReplace::print(std::ostream& os) const {
  os << "forallreplace(" << util::join(getPattern()) << ", " << util::join(getReplacement()) << ")";
}

std::ostream& operator<<(std::ostream& os, const ForAllReplace& forallreplace) {
  forallreplace.print(os);
  return os;
}

}