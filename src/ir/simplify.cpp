#include "ir/simplify.h"

#include <map>

#include "ir/ir.h"
#include "ir/ir_visitor.h"
#include "ir/ir_rewriter.h"
#include "taco/util/strings.h"
#include "taco/util/collections.h"
#include "taco/util/scopedmap.h"

using namespace std;

namespace taco {
namespace ir {

ir::Expr simplify(const ir::Expr& expr) {
  struct Rewriter : IRRewriter {
    void visit(const Add* op) {
      Expr a = rewrite(op->a);
      Expr b = rewrite(op->b);

      // 0 + b = b
      if (isa<Literal>(a)) {
        auto literal = to<Literal>(a);
        if (literal->type == ComponentType::Int && literal->value == 0) {
          expr = b;
          return;
        }
      }

      // a + 0 = a
      if (isa<Literal>(b)) {
        auto literal = to<Literal>(b);
        if (literal->type == ComponentType::Int && literal->value == 0) {
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
      if (isa<Literal>(a)) {
        auto literal = to<Literal>(a);
        if (literal->type == ComponentType::Int && literal->value == 0) {
          expr = literal;
          return;
        }
      }

      // a * 0 = 0
      if (isa<Literal>(b)) {
        auto literal = to<Literal>(b);
        if (literal->type == ComponentType::Int && literal->value == 0) {
          expr = literal;
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
  return Rewriter().rewrite(expr);
}

ir::Stmt simplify(const ir::Stmt& stmt) {
  // Perform copy propagation on variables that are added to a product of zero
  // and never re-assign, e.g. `int B1_pos = (0 * 42) + iB;`. These occur when
  // emitting code for top levels that are dense.

  // Collect candidates. These are variables that are never re-defined in the
  // scope they are declared in.
  struct CopyPropagationCandidates : IRVisitor {
    map<Stmt,Expr> varDeclsToRemove;
    util::ScopedMap<Expr,Stmt> declarations;

    using IRVisitor::visit;

    void visit(const Scope* scope) {
      declarations.scope();
      scope->scopedStmt.accept(this);
      declarations.unscope();
    }

    void visit(const VarAssign* assign) {
      if (assign->lhs.type() != ComponentType::Int) {
        return;
      }

      if (assign->is_decl) {
        Expr rhs = simplify(assign->rhs);
        if (isa<Var>(rhs)) {
          varDeclsToRemove.insert({assign, rhs});
          declarations.insert({assign->lhs, Stmt(assign)});
        }
      }
      else if (declarations.contains(assign->lhs)) {
        varDeclsToRemove.erase(declarations.get(assign->lhs));
      }
    }
  };
  CopyPropagationCandidates candidates;
  stmt.accept(&candidates);

  // Remove candidate var definitions and replace uses.
  struct CopyPropagation : IRRewriter {
    map<Stmt,Expr> varDeclsToRemove;
    util::ScopedMap<Expr,Expr> varsToReplace;

    void visit(const Scope* scope) {
      varsToReplace.scope();
      stmt = rewrite(scope->scopedStmt);
      varsToReplace.unscope();
    }

    void visit(const VarAssign* assign) {
      if (assign->is_decl && util::contains(varDeclsToRemove, Stmt(assign))) {
        varsToReplace.insert({assign->lhs, varDeclsToRemove.at(Stmt(assign))});
        stmt = Stmt();
        return;
      }
      IRRewriter::visit(assign);
    }

    void visit(const Var* var) {
      if (varsToReplace.contains(var)) {
        expr = varsToReplace.get(var);
        return;
      }
      IRRewriter::visit(var);
    }
  };
  CopyPropagation copyPropagation;
  copyPropagation.varDeclsToRemove = candidates.varDeclsToRemove;
  return copyPropagation.rewrite(stmt);
}

}}
