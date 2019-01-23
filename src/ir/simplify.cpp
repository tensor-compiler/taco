#include "taco/ir/simplify.h"

#include <map>
#include <queue>

#include "taco/ir/ir.h"
#include "taco/ir/ir_visitor.h"
#include "taco/ir/ir_rewriter.h"
#include "taco/util/strings.h"
#include "taco/util/collections.h"
#include "taco/util/scopedmap.h"

using namespace std;

namespace taco {
namespace ir {

struct ExpressionSimplifier : IRRewriter {
  using IRRewriter::visit;
  void visit(const Or* op) {
    Expr a = rewrite(op->a);
    Expr b = rewrite(op->b);

    // true || b = true
    // false || b = b
    if (isa<Literal>(a)) {
      auto literal = to<Literal>(a);
      expr = literal->getValue<bool>() ? a : b;
      return;
    }

    // a || true = true
    // a || false = a
    if (isa<Literal>(b)) {
      auto literal = to<Literal>(b);
      expr = literal->getValue<bool>() ? b : a;
      return;
    }

    if (a == op->a && b == op->b) {
      expr = op;
    }
    else {
      expr = Or::make(a, b);
    }
  }

  void visit(const And* op) {
    Expr a = rewrite(op->a);
    Expr b = rewrite(op->b);

    // true && b = b
    // false && b = false
    if (isa<Literal>(a)) {
      auto literal = to<Literal>(a);
      expr = literal->getValue<bool>() ? b : a;
      return;
    }

    // a && true = a
    // a && false = false
    if (isa<Literal>(b)) {
      auto literal = to<Literal>(b);
      expr = literal->getValue<bool>() ? a : b;
      return;
    }

    if (a == op->a && b == op->b) {
      expr = op;
    }
    else {
      expr = And::make(a, b);
    }
  }

  void visit(const Add* op) {
    Expr a = rewrite(op->a);
    Expr b = rewrite(op->b);

    // 1 + 1 = 2
    if (isa<Literal>(a) && isa<Literal>(b)) {
      auto lita = to<Literal>(a);
      auto litb = to<Literal>(b);
      auto typea = lita->type;
      auto typeb = litb->type;
      if (typea == typeb && isScalar(typea)) {
        if (typea.isInt()) {
          expr = Literal::make(lita->getIntValue()+litb->getIntValue(), typea);
          return;
        }
      }
    }

    // 0 + b = b
    if (isa<Literal>(a)) {
      auto literal = to<Literal>(a);
      if (literal->equalsScalar(0)) {
        expr = b;
        return;
      }
    }

    // a + 0 = a
    if (isa<Literal>(b)) {
      auto literal = to<Literal>(b);
      if (literal->equalsScalar(0)) {
        expr = a;
        return;
      }
    }

    if (a == op->a && b == op->b) {
      expr = op;
    }
    else {
      expr = Add::make(a, b);
    }
  }

  void visit(const Mul* op) {
    Expr a = rewrite(op->a);
    Expr b = rewrite(op->b);

    // 0 * b = 0
    // 1 * b = b
    if (isa<Literal>(a)) {
      auto literal = to<Literal>(a);
      if (literal->equalsScalar(0)) {
        expr = literal;
        return;
      }
      else if(literal->equalsScalar(1)) {
        expr = b;
        return;
      }
    }

    // a * 0 = 0
    // a * 1 = a
    if (isa<Literal>(b)) {
      auto literal = to<Literal>(b);

      if (literal->equalsScalar(0)) {
        expr = literal;
        return;
      }
      else if(literal->equalsScalar(1)) {
        expr = a;
        return;
      }
    }

    if (a == op->a && b == op->b) {
      expr = op;
    }
    else {
      expr = Mul::make(a, b);
    }
  }
};

ir::Expr simplify(const ir::Expr& expr) {
  return ExpressionSimplifier().rewrite(expr);
}

ir::Stmt simplify(const ir::Stmt& stmt) {
  // Perform copy propagation on variables that are added to a product of zero
  // and never re-assign, e.g. `int B1_pos = (0 * 42) + iB;`. These occur when
  // emitting code for top levels that are dense.

  // Collect candidates. These are variables that are never re-defined in the
  // scope they are declared in.
  struct CopyPropagationCandidates : IRVisitor {
    map<Stmt,Expr> varDeclsToRemove;
    multimap<Expr,Expr> dependencies;
    util::ScopedMap<Expr,Stmt> declarations;

    using IRVisitor::visit;

    void visit(const Scope* scope) {
      declarations.scope();
      scope->scopedStmt.accept(this);
      declarations.unscope();
    }

    void visit(const VarDecl* decl) {
      if (!decl->var.type().isInt()) {
        return;
      }

      Expr rhs = simplify(decl->rhs);
      if (isa<Var>(rhs)) {
        varDeclsToRemove.insert({decl, rhs});
        dependencies.insert({rhs, decl->var});
        declarations.insert({decl->var, Stmt(decl)});
      }
    }

    void visit(const Assign* assign) {
      if (!assign->lhs.type().isInt()) {
        return;
      }
      
      queue<Expr> invalidVars;
      invalidVars.push(assign->lhs);

      while (!invalidVars.empty()) {
        Expr invalidVar = invalidVars.front();
        invalidVars.pop();

        if (declarations.contains(invalidVar)) {
          varDeclsToRemove.erase(declarations.get(invalidVar));
        }

        auto range = dependencies.equal_range(invalidVar);
        for (auto dep = range.first; dep != range.second; ++dep) {
          invalidVars.push(dep->second);
        }
      }
    }
  };

  CopyPropagationCandidates candidates;
  stmt.accept(&candidates);

  // Copy propagation (remove candidate var definitions and replace uses) and
  // expression simplification.
  struct Simplifier : ExpressionSimplifier {
    map<Stmt,Expr> varDeclsToRemove;
    util::ScopedMap<Expr,Expr> varsToReplace;

    using ExpressionSimplifier::visit;

    void visit(const Scope* scope) {
      varsToReplace.scope();
      stmt = rewrite(scope->scopedStmt);
      varsToReplace.unscope();
    }

    void visit(const VarDecl* decl) {
      if (util::contains(varDeclsToRemove, Stmt(decl))) {
        varsToReplace.insert({decl->var, varDeclsToRemove.at(Stmt(decl))});
        stmt = Stmt();
        return;
      }
      IRRewriter::visit(decl);
    }

    void visit(const Var* var) {
      bool replaced = false;
      Expr cvar = var;
      while (varsToReplace.contains(cvar)) {
        cvar = varsToReplace.get(cvar);
        replaced = true;
      }
      if (replaced) {
        expr = cvar;
        return;
      }
      IRRewriter::visit(var);
    }
  };
  Simplifier copyPropagation;
  copyPropagation.varDeclsToRemove = candidates.varDeclsToRemove;
  return copyPropagation.rewrite(stmt);
}

}}
