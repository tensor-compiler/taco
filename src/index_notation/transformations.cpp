#include "taco/index_notation/transformations.h"

#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_rewriter.h"
#include "taco/index_notation/index_notation_nodes.h"

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

bool Transformation::isValid(IndexStmt stmt, string* reason) const {
  return transformation->isValid(stmt, reason);
}

IndexStmt Transformation::apply(IndexStmt stmt) const {
  return transformation->apply(stmt);
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

bool Reorder::isValid(IndexStmt stmt, std::string* reason) const {
  INIT_REASON(reason);
  string r;

  // Must be concrete notation
  if (!isConcreteNotation(stmt, &r)) {
    *reason = "The index statement is not valid concrete index notation.\n" + r;
    return false;
  }

  IndexVar i = this->geti();
  IndexVar j = this->getj();

  // Variables can be reorderd if their forall statements are directly nested,
  // and the inner forall's statement is an assignment with an associative
  // compound operator.
  bool valid = false;
  match(stmt,
    function<void(const ForallNode*,Matcher*)>([&](const ForallNode* node,
                                                   Matcher* ctx) {
      Forall foralli(node);

      if ((foralli.getIndexVar() == i || foralli.getIndexVar() == j) &&
          isa<Forall>(foralli.getStmt())) {
        if (foralli.getIndexVar() == j) {
          swap(i, j);
        }
        auto forallj = to<Forall>(foralli.getStmt());
        // TODO: Check that all compound assignments in forallj.getStmt() are
        //       associative.
        if (forallj.getIndexVar() == j) {
          valid = true;
          return;
        }
      }

      ctx->match(foralli.getStmt());
    })
  );

  if (!valid) {
    *reason = "Does not contain two consecutive nested loops.";
  }

  return valid;
}

IndexStmt Reorder::apply(IndexStmt stmt) const {
  taco_iassert(isValid(stmt));

  struct ReorderRewriter : public IndexNotationRewriter {
    using IndexNotationRewriter::visit;

    Reorder reorder;
    ReorderRewriter(Reorder reorder) : reorder(reorder) {}

    void visit(const ForallNode* node) {
      Forall foralli(node);
      taco_iassert(isa<Forall>(foralli.getStmt()));

      IndexVar i = reorder.geti();
      IndexVar j = reorder.getj();

      // Nested loops with assignment or associative compound assignment.
      if (foralli.getIndexVar() == i || foralli.getIndexVar() == j) {
        if (foralli.getIndexVar() == j) {
          swap(i, j);
        }
        auto forallj = to<Forall>(foralli.getStmt());
        taco_iassert(forallj.getIndexVar() == j);
        stmt = forall(j, forall(i, forallj.getStmt()));
        return;
      }

      IndexNotationRewriter::visit(node);
    }
  };
  ReorderRewriter rewriter(*this);
  return rewriter.rewrite(stmt);
}

void Reorder::print(std::ostream& os) const {
  os << "reorder(" << geti() << ", " << getj() << ")";
}

std::ostream& operator<<(std::ostream& os, const Reorder& reorder) {
  reorder.print(os);
  return os;
}


// class P
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

bool Precompute::isValid(IndexStmt stmt, std::string* reason) const {
  INIT_REASON(reason);

  IndexExpr expr = getExpr();
  Assignment assignment = getAssignmentContainingExpr(stmt, expr);

  // There must be one assignment with the expr.
  if (!assignment.defined()) {
    return false;
  }

  // Every operator in expr must be associative.
  // TODO: Check this generally

  // What exactly are the distribution rules?
  /*
  IndexExpr op = assignment.getOperator();
  if (op.defined()) {
    // TODO: Check distribution generally.  For now just checking that expr
    //       operators are mults if assignment operator is add.
    if (!isa<Add>(op)) {
      return false;
    }
    match(expr,
        function<void(const BinaryExprNode*,Matcher*)>([&](
            const BinaryExprNode* node, Matcher* ctx) {
        })
  );
  }
  */

  return true;
}

IndexStmt Precompute::apply(IndexStmt stmt) const {
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

        IndexStmt consumer = forall(i, replace(s, {{e, ws(i)}}, {}));
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

}
