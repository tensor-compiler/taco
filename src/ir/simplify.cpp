#include "ir/simplify.h"

#include <map>

#include "ir/ir.h"
#include "ir/ir_visitor.h"
#include "ir/ir_rewriter.h"
#include "taco/util/collections.h"

using namespace std;

namespace taco {
namespace ir {

ir::Expr simplify(const ir::Expr& expr) {
  struct Rewriter : IRRewriter {
    using IRRewriter::visit;
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

  // Collect variables to replace
  struct Visitor : IRVisitor {
    using IRVisitor::visit;
    map<Expr,Expr> varsToReplace;

    void visit(const VarAssign* op) {
      if (op->is_decl) {
        taco_iassert(!util::contains(varsToReplace, op->lhs)) <<
            op->lhs << " is declared twice";

        Expr rhs = simplify(op->rhs);
        if (rhs.as<Var>() != nullptr) {
          std::cout << Stmt(op) << std::endl;
          taco_iassert(op->lhs.as<Var>() != nullptr);
          varsToReplace.insert({op->lhs,rhs});
        }
      }
      // Check that this assignment doesn't re-assign to any of the candidate
      // vars to replace. If it does then remove it as a candidate.
      else if (util::contains(varsToReplace, op->lhs)) {
        varsToReplace.erase(op->lhs);
      }
    }
  };
  Visitor visitor;
  stmt.accept(&visitor);

  // Remove definitions and replace uses
  struct Rewriter : IRRewriter {
    using IRRewriter::visit;
    map<Expr,Expr> varsToReplace;

    void visit(const VarAssign* op) {
      if (util::contains(varsToReplace, op->lhs)) {
        stmt = Stmt();
      }
      else {
        IRRewriter::visit(op);
      }
    }

    void visit(const Var* op) {
      if (util::contains(varsToReplace, Expr(op))) {
        expr = varsToReplace.at(op);
      }
      else {
        expr = op;
      }
    }
  };
  Rewriter rewriter;
  rewriter.varsToReplace = visitor.varsToReplace;
  return rewriter.rewrite(stmt);
}

}}
