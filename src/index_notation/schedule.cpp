#include "taco/index_notation/schedule.h"

#include <map>

#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_rewriter.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/util/collections.h"
#include "taco/util/strings.h"
#include "taco/error/error_messages.h"

using namespace std;

namespace taco {

// class Reorder
struct Reorder::Content {
  IndexVar i;
  IndexVar j;
};

Reorder::Reorder() : content(nullptr) {
}

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

IndexStmt Reorder::apply(IndexStmt stmt) {
  taco_iassert(isValid(stmt));

  struct ReorderRewriter : public IndexNotationRewriter {
    using IndexNotationRewriter::visit;

    Reorder reorder;

    void visit(const ForallNode* node) {
      Forall foralli(node);

      IndexVar i = reorder.geti();
      IndexVar j = reorder.getj();

      // Nested loops with assignment or associative compound assignment.
      // TODO: Add associative test
      if ((foralli.getIndexVar() == i || foralli.getIndexVar() == j) &&  isa<Forall>(foralli.getStmt())) {
        if (foralli.getIndexVar() == j) {
          swap(i, j);
        }
        auto forallj = to<Forall>(foralli.getStmt());
        if (forallj.getIndexVar() == j && isa<Assignment>(forallj.getStmt())) {
          stmt = forall(j, forall(i, forallj.getStmt()));
          return;
        }
      }

      IndexNotationRewriter::visit(node);
    }
  };
  ReorderRewriter rewriter;
  rewriter.reorder = *this;
  return rewriter.rewrite(stmt);
}

bool Reorder::isValid(IndexStmt stmt, std::string* reason) {
  INIT_REASON(reason);
  string r;

  // Must be concrete notation
  if (!isConcreteNotation(stmt, &r)) {
    *reason = "The index statement is not valid concrete index notation\n" + r;
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

      // TODO: Add associative test
      if ((foralli.getIndexVar() == i || foralli.getIndexVar() == j) &&
          isa<Forall>(foralli.getStmt())) {
        if (foralli.getIndexVar() == j) {
          swap(i, j);
        }
        auto forallj = to<Forall>(foralli.getStmt());
        if (forallj.getIndexVar() == j && isa<Assignment>(forallj.getStmt())) {
          valid = true;
          return;
        }
      }

      ctx->match(foralli.getStmt());
    })
  );

  return valid;
}

std::ostream& operator<<(std::ostream& os, const Reorder& reorder) {
  return os << "reorder(" << reorder.geti() << ", " << reorder.getj() << ")";
}


// class Workspace
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

bool Precompute::isValid(IndexStmt stmt, std::string* reason) {
  INIT_REASON(reason);
  return false;
}

IndexStmt Precompute::apply(IndexStmt stmt) {

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

bool Precompute::defined() const {
  return content != nullptr;
}

std::ostream& operator<<(std::ostream& os, const Precompute& workspace) {
  return os << workspace.getExpr() << ": " << workspace.geti() << ", "
            << workspace.getiw() << ", " << workspace.getWorkspace();
}


// class Schedule
struct Schedule::Content {
  map<IndexExpr, Precompute> precomputes;
};

Schedule::Schedule() : content(new Content) {
}

std::vector<Precompute> Schedule::getPrecomputes() const {
  vector<Precompute> workspaces;
  for (auto& workspace : content->precomputes) {
    workspaces.push_back(workspace.second);
  }
  return workspaces;
}

Precompute Schedule::getPrecompute(IndexExpr expr) const {
  if (!util::contains(content->precomputes, expr)) {
    return Precompute();
  }
  return content->precomputes.at(expr);
}

void Schedule::addPrecompute(Precompute workspace) {
  if (!util::contains(content->precomputes, workspace.getExpr())) {
    content->precomputes.insert({workspace.getExpr(), workspace});
  }
  else {
    content->precomputes.at(workspace.getExpr()) = workspace;
  }
}

void Schedule::clearPrecomputes() {
  content->precomputes.clear();
}

std::ostream& operator<<(std::ostream& os, const Schedule& schedule) {
  auto workspaces = schedule.getPrecomputes();
  if (workspaces.size() > 0) {
    os << "Workspace Commands:" << endl << util::join(workspaces, "\n");
  }
  return os;
}

}
