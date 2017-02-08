#include "available_exprs.h"

#include <stack>
#include <set>

#include "var.h"
#include "expr_visitor.h"
#include "expr_nodes.h"
#include "util/collections.h"

using namespace std;
using namespace taco::internal;

namespace taco {
namespace lower {

/// Retrieves the available sub-expression at the index variable
vector<taco::Expr> getAvailableExpressions(const taco::Expr& expr,
                                           const std::vector<taco::Var>& vars) {

  // Available expressions are the maximal sub-expressions that only contain
  // operands whose index variables have all been visited.
  struct ExtractAvailableExpressions : public internal::ExprVisitor {
    Var var;
    set<Var> visitedVars;

    /// A vector of all the available expressions
    vector<Expr> availableExpressions;

    /// A stack of active expressions and a bool saying whether they are
    /// available. Expressions are moved from the stack to availableExpressions
    /// when an inactive sub-expression is found.
    stack<pair<Expr,bool>> activeExpressions;

    vector<Expr> get(const Expr& expr, const vector<Var>& vars) {
      this->visitedVars = set<Var>(vars.begin(), vars.end());
      this->var = var;

      expr.accept(this);

      // Take out available expressions that are just immediates or a scalars.
      // No point in storing these to a temporary.
      // TODO ...

      return availableExpressions;
    }

    void visit(const internal::Read* op) {
      bool available = true;
      for (auto& var : op->indexVars) {
        if (!util::contains(visitedVars, var)) {
          available = false;
          break;
        }
      }
      activeExpressions.push({op, available});
    }

    void visit(const internal::UnaryExpr* op) {
      op->a.accept(this);
      iassert(activeExpressions.size() == 1);

      pair<Expr,bool> a = activeExpressions.top();
      activeExpressions.pop();

      activeExpressions.push({op, a.second});
    }

    void visit(const BinaryExpr* op) {
      op->a.accept(this);
      op->b.accept(this);
      iassert(activeExpressions.size() == 2);

      pair<Expr,bool> a = activeExpressions.top();
      activeExpressions.pop();
      pair<Expr,bool> b = activeExpressions.top();
      activeExpressions.pop();

      if (a.second && b.second) {
        activeExpressions.push({op, true});
      }
      else {
        if (a.second) {
          availableExpressions.push_back(a.first);
        }
        if (b.second) {
          availableExpressions.push_back(b.first);
        }
        activeExpressions.push({op, false});
      }
    }

    // Immediates are always available (can compute them anywhere)
    void visit(const internal::ImmExpr* op) {
      activeExpressions.push({op,true});
    }
  };

  return ExtractAvailableExpressions().get(expr, vars);
}

}}
