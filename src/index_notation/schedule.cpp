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

    void visit(const ForallNode* node) {
      Forall foralli = node;

      // Nested loops with assignment or associative compound assignment.
      // TODO: Add associative test
      if (isa<Forall>(foralli.getStmt())) {
        auto forallj = to<Forall>(foralli.getStmt());
        if (isa<Assignment>(forallj.getStmt())) {
          auto s = forallj.getStmt();
          auto i = foralli.getIndexVar();
          auto j = forallj.getIndexVar();
          stmt = forall(j, forall(i, s));
          return;
        }
      }

      stmt = foralli;
    }
  };
  stmt = ReorderRewriter().rewrite(stmt);

  return stmt;
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
      Forall foralli = node;
      // TODO: Add associative test
      if ((foralli.getIndexVar() == i || foralli.getIndexVar() == j) &&
          isa<Forall>(foralli.getStmt())) {
        if (foralli.getIndexVar() == j) {
          swap(i,j);
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
struct Workspace::Content {
  IndexExpr expr;
  IndexVar i;
  IndexVar iw;
  TensorVar workspace;
};

Workspace::Workspace() : content(nullptr) {
}

Workspace::Workspace(IndexExpr expr, IndexVar i, IndexVar iw,
                     TensorVar workspace) : content(new Content) {
  content->expr = expr;
  content->i = i;
  content->iw = iw;
  content->workspace = workspace;
}

IndexExpr Workspace::getExpr() const {
  return content->expr;
}

IndexVar Workspace::geti() const {
  return content->i;
}

IndexVar Workspace::getiw() const {
  return content->iw;
}

TensorVar Workspace::getWorkspace() const {
  return content->workspace;
}

bool Workspace::isValid(IndexStmt stmt, std::string* reason) {
  INIT_REASON(reason);

  return false;
}

IndexStmt Workspace::apply(IndexStmt stmt) {
  return stmt;
}

bool Workspace::defined() const {
  return content != nullptr;
}

std::ostream& operator<<(std::ostream& os, const Workspace& workspace) {
  return os << workspace.getExpr() << ": " << workspace.geti() << ", "
            << workspace.getiw() << ", " << workspace.getWorkspace();
}


// class Schedule
struct Schedule::Content {
  map<IndexExpr, Workspace> workspaces;
};

Schedule::Schedule() : content(new Content) {
}

std::vector<Workspace> Schedule::getWorkspaces() const {
  vector<Workspace> workspaces;
  for (auto& workspace : content->workspaces) {
    workspaces.push_back(workspace.second);
  }
  return workspaces;
}

Workspace Schedule::getWorkspace(IndexExpr expr) const {
  if (!util::contains(content->workspaces, expr)) {
    return Workspace();
  }
  return content->workspaces.at(expr);
}

void Schedule::addWorkspace(Workspace workspace) {
  if (!util::contains(content->workspaces, workspace.getExpr())) {
    content->workspaces.insert({workspace.getExpr(), workspace});
  }
  else {
    content->workspaces.at(workspace.getExpr()) = workspace;
  }
}

void Schedule::clearWorkspaces() {
  content->workspaces.clear();
}

std::ostream& operator<<(std::ostream& os, const Schedule& schedule) {
  auto workspaces = schedule.getWorkspaces();
  if (workspaces.size() > 0) {
    os << "Workspace Commands:" << endl << util::join(workspaces, "\n");
  }
  return os;
}

}
