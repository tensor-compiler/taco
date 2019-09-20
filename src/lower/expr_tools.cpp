#include "expr_tools.h"

#include <stack>
#include <set>

#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/index_notation/index_notation_visitor.h"
#include "taco/util/collections.h"

using namespace std;

namespace taco {

/// Retrieves the available sub-expression at the index variable
vector<IndexExpr> getAvailableExpressions(const IndexExpr& expr,
                                          const vector<IndexVar>& vars) {

  // Available expressions are the maximal sub-expressions that only contain
  // operands whose index variables have all been visited.
  struct ExtractAvailableExpressions : public IndexNotationVisitor {
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
      //this->var = var;

      expr.accept(this);

      taco_iassert(activeExpressions.size() == 1);
      if (activeExpressions.top().second) {
        availableExpressions.push_back(activeExpressions.top().first);
      }

      // Take out available expressions that are just literals or a scalars.
      // No point in storing these to a temporary.
      // TODO ...

      return availableExpressions;
    }

    using IndexNotationVisitor::visit;

    void visit(const AccessNode* op) {
      bool available = true;
      for (auto& var : op->indexVars) {
        if (!util::contains(visitedVars, var)) {
          available = false;
          break;
        }
      }
      activeExpressions.push({op, available});
    }

    // Literals are always available and can be computed anywhere
    void visit(const LiteralNode* op) {
      activeExpressions.push({op,true});
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
  };

  return ExtractAvailableExpressions().get(expr, vars);
}

IndexExpr getSubExprOld(IndexExpr expr, const vector<IndexVar>& vars) {
  class SubExprVisitor : public IndexNotationVisitor {
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

    using IndexNotationVisitorStrict::visit;

    void visit(const AccessNode* op) {
      // If any variable is in the set of index variables, then the expression
      // has not been emitted at a previous level, so we keep it.
      for (auto& indexVar : op->indexVars) {
        if (util::contains(vars, indexVar)) {
          subExpr = op;
          return;
        }
      }
      subExpr = IndexExpr();
    }

    void visit(const LiteralNode* op) {
      subExpr = IndexExpr();
    }

    void visit(const UnaryExprNode* op) {
      IndexExpr a = getSubExpression(op->a);
      if (a.defined()) {
        subExpr = op;
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
  };
  return SubExprVisitor(vars).getSubExpression(expr);
}

class SubExprVisitor : public IndexExprVisitorStrict {
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

  using IndexExprVisitorStrict::visit;

  void visit(const AccessNode* op) {
    // If any variable is in the set of index variables, then the expression
    // has not been emitted at a previous level, so we keep it.
    for (auto& indexVar : op->indexVars) {
      if (util::contains(vars, indexVar)) {
        subExpr = op;
        return;
      }
    }
    subExpr = IndexExpr();
  }

  void visit(const LiteralNode* op) {
    subExpr = IndexExpr();
  }

  template <class T>
  IndexExpr unarySubExpr(const T* op) {
    IndexExpr a = getSubExpression(op->a);
    return a.defined() ? op : IndexExpr();
  }

  void visit(const NegNode* op) {
    subExpr = unarySubExpr(op);
  }

  void visit(const SqrtNode* op) {
    subExpr = unarySubExpr(op);
  }

  template <class T>
  IndexExpr binarySubExpr(const T* op) {
    IndexExpr a = getSubExpression(op->a);
    IndexExpr b = getSubExpression(op->b);
    if (a.defined() && b.defined()) {
      return new T(a, b);
    }
    else if (a.defined()) {
      return a;
    }
    else if (b.defined()) {
      return b;
    }

    return IndexExpr();
  }

  void visit(const AddNode* op) {
    subExpr = binarySubExpr(op);
  }

  void visit(const SubNode* op) {
    subExpr = binarySubExpr(op);
  }

  void visit(const MulNode* op) {
    subExpr = binarySubExpr(op);
  }

  void visit(const DivNode* op) {
    subExpr = binarySubExpr(op);
  }

  void visit(const CastNode* op) {
    taco_not_supported_yet;
  }

  void visit(const CallIntrinsicNode* op) {
    taco_not_supported_yet;
  }

  void visit(const ReductionNode* op) {
    subExpr = op;
  }

  void visit(const AssignmentNode* op) {
    taco_ierror;
  }

  void visit(const ForallNode* op) {
    taco_ierror;
  }

  void visit(const WhereNode* op) {
    taco_ierror;
  }
};

IndexExpr getSubExpr(IndexExpr expr, const vector<IndexVar>& vars) {
  return SubExprVisitor(vars).getSubExpression(expr);
}

}
