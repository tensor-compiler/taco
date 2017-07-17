#include "expr_tools.h"

#include <stack>
#include <set>

#include "taco/expr.h"
#include "taco/expr_nodes/expr_nodes.h"
#include "taco/expr_nodes/expr_visitor.h"
#include "taco/util/collections.h"

using namespace std;
using namespace taco::expr_nodes;

namespace taco {
namespace lower {

/// Retrieves the available sub-expression at the index variable
vector<IndexExpr> getAvailableExpressions(const IndexExpr& expr,
                                          const vector<IndexVar>& vars) {

  // Available expressions are the maximal sub-expressions that only contain
  // operands whose index variables have all been visited.
  struct ExtractAvailableExpressions : public expr_nodes::ExprVisitor {
    IndexVar var;
    set<IndexVar> visitedVars;

    /// A vector of all the available expressions
    vector<IndexExpr> availableExpressions;

    /// A stack of active expressions and a bool saying whether they are
    /// available. Expressions are moved from the stack to availableExpressions
    /// when an inactive sub-expression is found.
    stack<pair<IndexExpr,bool>> activeExpressions;

    vector<IndexExpr> get(const IndexExpr& expr, const vector<IndexVar>& vars) {
      this->visitedVars = set<IndexVar>(vars.begin(), vars.end());
      this->var = var;

      expr.accept(this);

      // Take out available expressions that are just immediates or a scalars.
      // No point in storing these to a temporary.
      // TODO ...

      return availableExpressions;
    }

    using expr_nodes::ExprVisitor::visit;

    void visit(const ReadNode* op) {
      bool available = true;
      for (auto& var : op->indexVars) {
        if (!util::contains(visitedVars, var)) {
          available = false;
          break;
        }
      }
      activeExpressions.push({op, available});
    }

    void visit(const UnaryExprNode* op) {
      op->a.accept(this);
      taco_iassert(activeExpressions.size() >= 1);

      pair<IndexExpr,bool> a = activeExpressions.top();
      activeExpressions.pop();

      activeExpressions.push({op, a.second});
    }

    void visit(const BinaryExprNode* op) {
      op->a.accept(this);
      op->b.accept(this);
      taco_iassert(activeExpressions.size() >= 2);

      pair<IndexExpr,bool> a = activeExpressions.top();
      activeExpressions.pop();
      pair<IndexExpr,bool> b = activeExpressions.top();
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
    void visit(const ImmExprNode* op) {
      activeExpressions.push({op,true});
    }
  };

  return ExtractAvailableExpressions().get(expr, vars);
}

IndexExpr getSubExpr(IndexExpr expr, const vector<IndexVar>& vars) {
  class SubExprVisitor : public ExprVisitor {
  public:
    SubExprVisitor(const vector<IndexVar>& vars) {
      this->vars.insert(vars.begin(), vars.end());
    }

    IndexExpr getSubExpression(const IndexExpr& expr) {
      visit(expr);
      IndexExpr e = subExpr;
      subExpr = IndexExpr();
      return e;
    }

  private:
    set<IndexVar> vars;
    IndexExpr     subExpr;

    using ExprVisitorStrict::visit;

    void visit(const ReadNode* op) {
      for (auto& indexVar : op->indexVars) {
        if (util::contains(vars, indexVar)) {
          subExpr = op;
          return;
        }
      }
      subExpr = IndexExpr();
    }

    void visit(const UnaryExprNode* op) {
      IndexExpr a = getSubExpression(op->a);
      if (a.defined()) {
        subExpr = a;
      }
      else {
        subExpr = IndexExpr();
      }
    }

    void visit(const BinaryExprNode* op) {
      IndexExpr a = getSubExpression(op->a);
      IndexExpr b = getSubExpression(op->b);
      if (a.defined() && b.defined()) {
        subExpr = op;
      }
      else if (a.defined()) {
        subExpr = a;
      }
      else if (b.defined()) {
        subExpr = b;
      }
      else {
        subExpr = IndexExpr();
      }
    }

    void visit(const ImmExprNode* op) {
      subExpr = IndexExpr();
    }

  };
  return SubExprVisitor(vars).getSubExpression(expr);
}

}}
