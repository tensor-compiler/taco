#include "taco/ir/simplify.h"

#include <map>
#include <queue>

#include "taco/ir/ir.h"
#include "taco/ir/ir_visitor.h"
#include "taco/ir/ir_rewriter.h"
#include "taco/util/strings.h"
#include "taco/util/collections.h"
#include "taco/util/scopedmap.h"

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
      auto resulttype = max_type(typea, typeb);
      if(isScalar(typea) && isScalar(typeb) && (resulttype.isInt() || resulttype.isUInt())) {
        // convert literals to result type uint -> int
        auto litaval = lita->getTypedVal();
        auto litbval = litb->getTypedVal();
        if (typea != resulttype) {
          litaval = TypedComponentVal(resulttype, (int) litaval.getAsIndex());
        }
        if (typeb != resulttype) {
          litbval = TypedComponentVal(resulttype, (int) litbval.getAsIndex());
        }
        expr = Literal::make(litaval + litbval, resulttype);
        return; 
      }
    }

    // (c + d) + b = c + (d + b)
    // TODO: handle operands of different types
    if (isa<Add>(a) && isa<Literal>(to<Add>(a)->b) && isa<Literal>(b)) {
      auto adda = to<Add>(a);
      auto litd = to<Literal>(adda->b);
      auto litb = to<Literal>(b);
      auto typec = adda->a.type();
      auto typed = litd->type;
      auto typeb = litb->type;
      if (typec == typed && typed == typeb && isScalar(typeb) && (typeb.isInt() || typeb.isUInt())){
        auto litdval = litd->getTypedVal();
        auto litbval = litb->getTypedVal();
        expr = simplify(Add::make(adda->a, Literal::make(litdval + litbval, typeb)));
        return;
      }
    }

    // (c - d) + b = c + (b - d)
    // TODO: handle operands of different types
    if (isa<Sub>(a) && isa<Literal>(to<Sub>(a)->b) && isa<Literal>(b)) {
      auto suba = to<Sub>(a);
      auto litd = to<Literal>(suba->b);
      auto litb = to<Literal>(b);
      auto typec = suba->a.type();
      auto typed = litd->type;
      auto typeb = litb->type;
      if (typec == typed && typed == typeb && isScalar(typeb) && (typeb.isInt() || typeb.isUInt())){
        auto litdval = litd->getTypedVal();
        auto litbval = litb->getTypedVal();
        expr = simplify(Add::make(suba->a, Literal::make(litbval - litdval, typeb)));
        return;
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

  void visit(const Sub* op) {
    Expr a = rewrite(op->a);
    Expr b = rewrite(op->b);

    // 2 - 1 = 1
    if (isa<Literal>(a) && isa<Literal>(b)) {
      auto lita = to<Literal>(a);
      auto litb = to<Literal>(b);
      auto typea = lita->type;
      auto typeb = litb->type;
      auto resulttype = max_type(typea, typeb);
      if(isScalar(typea) && isScalar(typeb) && (resulttype.isInt() || resulttype.isUInt())) {
        // convert literals to result type uint -> int
        auto litaval = lita->getTypedVal();
        auto litbval = litb->getTypedVal();
        if (typea != resulttype) {
          litaval = TypedComponentVal(resulttype, (int) litaval.getAsIndex());
        }
        if (typeb != resulttype) {
          litbval = TypedComponentVal(resulttype, (int) litbval.getAsIndex());
        }
        expr = Literal::make(litaval - litbval, resulttype);
        return; 
      }
    }

    // 0 - b = -b
    if (isa<Literal>(a)) {
      auto literal = to<Literal>(a);
      if (literal->equalsScalar(0)) {
        expr = Neg::make(b);
        return;
      }
    }

    // a - 0 = a
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
      expr = Sub::make(a, b);
    }
  }

  void visit(const Mul* op) {
    Expr a = rewrite(op->a);
    Expr b = rewrite(op->b);

    // a * b = ab
    if (isa<Literal>(a) && isa<Literal>(b)) {
      auto lita = to<Literal>(a);
      auto litb = to<Literal>(b);
      auto typea = lita->type;
      auto typeb = litb->type;
      auto resulttype = max_type(typea, typeb);
      if(isScalar(typea) && isScalar(typeb) && (resulttype.isInt() || resulttype.isUInt())) {
        // convert literals to result type uint -> int
        auto litaval = lita->getTypedVal();
        auto litbval = litb->getTypedVal();
        if (typea != resulttype) {
          litaval = TypedComponentVal(resulttype, (int) litaval.getAsIndex());
        }
        if (typeb != resulttype) {
          litbval = TypedComponentVal(resulttype, (int) litbval.getAsIndex());
        }
        expr = Literal::make(litaval * litbval, resulttype);
        return; 
      }
    }

    // (c * d) * b = c * db
    if (isa<Mul>(a) && isa<Literal>(to<Mul>(a)->b) && isa<Literal>(b)) {
      auto mula = to<Mul>(a);
      auto litd = to<Literal>(mula->b);
      auto litb = to<Literal>(b);
      auto typec = mula->a.type();
      auto typed = litd->type;
      auto typeb = litb->type;
      if (typec == typed && typed == typeb && isScalar(typeb) && (typeb.isInt() || typeb.isUInt())){
        auto litdval = litd->getTypedVal();
        auto litbval = litb->getTypedVal();
        expr = simplify(Mul::make(mula->a, Literal::make(litdval * litbval, typeb)));
        return;
      }
    }

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

  void visit(const Div* op) {
    Expr a = rewrite(op->a);
    Expr b = rewrite(op->b);

    if (isa<Literal>(a) && isa<Literal>(b)) {
      auto lita = to<Literal>(a);
      auto litb = to<Literal>(b);
      auto typea = lita->type;
      auto typeb = litb->type;
      if (typea == typeb && isScalar(typea)) {
        if (typea.isInt()) {
          expr = Literal::make(lita->getIntValue()/litb->getIntValue(), typea);
          return;
        }
        else if (typea.isUInt()) {
          expr = Literal::make(lita->getUIntValue()/litb->getUIntValue(), typea);
          return;
        }
      }
    }

    // a / 1 = a
    if (isa<Literal>(b)) {
      auto literal = to<Literal>(b);

      if (literal->equalsScalar(1)) {
        expr = a;
        return;
      }
    }

    if (a == op->a && b == op->b) {
      expr = op;
    }
    else {
      expr = Div::make(a, b);
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

  // Identify all loop dependent variables. (This analysis is imprecise.)
  struct FindLoopDependentVars : IRVisitor {
    std::set<Expr> loopDependentVars;
    std::map<Expr,int> defLevel;
    int loopLevel = 0;

    using IRVisitor::visit;

    void visit(const For* op) {
      loopDependentVars.insert(op->var);
      loopLevel++;
      op->contents.accept(this);
      loopLevel--;
    }

    void visit(const While* op) {
      loopLevel++;
      op->contents.accept(this);
      loopLevel--;
    }

    void visit(const VarDecl* op) {
      defLevel.insert({op->var, loopLevel});
    }

    void visit(const Assign* op) {
      if (util::contains(defLevel, op->lhs) && 
          defLevel.at(op->lhs) < loopLevel) {
        loopDependentVars.insert(op->lhs);
      }
    }
  };

  FindLoopDependentVars findLoopDepVars;
  stmt.accept(&findLoopDepVars);

  // Copy propagation (remove candidate var definitions and replace uses) and
  // expression simplification. Also identify non-redundant variable 
  // declarations.
  // TODO: Currently does not handle the following pattern:
  //   b = a
  //   if (...) {
  //     ... = b
  //     b = d
  //   } else {
  //     ... = b
  //   }
  //   ... = b
  // The use of `b` in the if branch would be rewritten to `a` but not the use
  // in the else branch. To fix this, the visit method for if and case
  // statements would probably need to be overridden so that each branch is
  // evaluated starting with the same `varsToReplace`.
  struct Simplifier : ExpressionSimplifier {
    util::ScopedMap<Expr,std::pair<Expr,Stmt>> varsToReplace;
    std::set<Stmt> necessaryDecls;
    std::multimap<Expr,Expr> dependencies;
    util::ScopedMap<Expr,Stmt> declarations;
    std::set<Expr> loopDependentVars;

    using ExpressionSimplifier::visit;

    Simplifier(const std::set<Expr>& loopDependentVars) : 
        loopDependentVars(loopDependentVars) {}

    void visit(const Scope* scope) {
      declarations.scope();
      varsToReplace.scope();
      stmt = rewrite(scope->scopedStmt);
      varsToReplace.unscope();
      declarations.unscope();
    }

    void visit(const VarDecl* decl) {
      Expr rhs = rewrite(decl->rhs);
      stmt = (rhs == decl->rhs) ? decl : VarDecl::make(decl->var, rhs);

      declarations.insert({decl->var, stmt});
      if (decl->var.type().isInt() && isa<Var>(rhs) && 
          !util::contains(loopDependentVars, decl->var)) {
        taco_iassert(!varsToReplace.contains(decl->var)) 
            << "Copy propagation pass currently does not support variables " 
            << "with same name declared in nested scopes";
        varsToReplace.insert({decl->var, {rhs, stmt}});
        dependencies.insert({rhs, decl->var});
      }
    }

    void visit(const Assign* assign) {
      Expr lhs = isa<Var>(assign->lhs) ? assign->lhs : rewrite(assign->lhs);
      Expr rhs = rewrite(assign->rhs);
      stmt = (lhs == assign->lhs && rhs == assign->rhs) ? assign : 
             Assign::make(lhs, rhs);
      
      if (declarations.contains(lhs)) {
        taco_iassert(isa<Var>(lhs));
        necessaryDecls.insert(declarations.get(lhs));
      }

      if (!assign->lhs.type().isInt()) {
        return;
      }
      
      std::queue<Expr> invalidVars;
      invalidVars.push(assign->lhs);

      while (!invalidVars.empty()) {
        Expr invalidVar = invalidVars.front();
        invalidVars.pop();

        if (varsToReplace.contains(invalidVar)){
          varsToReplace.remove(invalidVar);
        }

        auto range = dependencies.equal_range(invalidVar);
        for (auto dep = range.first; dep != range.second; ++dep) {
          invalidVars.push(dep->second);
        }
      }
    }

    void visit(const Var* var) {
      expr = varsToReplace.contains(var) ? varsToReplace.get(var).first : var;
      if (declarations.contains(expr)) {
        necessaryDecls.insert(declarations.get(expr));
      }
    }
  };

  Simplifier copyPropagation(findLoopDepVars.loopDependentVars);
  Stmt simplifiedStmt = copyPropagation.rewrite(stmt);

  // Remove redundant variable declarations.
  struct RemoveRedundantStmts : IRRewriter {
    std::set<Stmt> necessaryDecls;
    
    using IRRewriter::visit;

    RemoveRedundantStmts(const std::set<Stmt>& necessaryDecls) : 
        necessaryDecls(necessaryDecls) {}

    void visit(const VarDecl* decl) {
      stmt = util::contains(necessaryDecls, decl)? decl : Block::make();
    }
  };

  simplifiedStmt = RemoveRedundantStmts(copyPropagation.necessaryDecls).rewrite(
      simplifiedStmt);
  return simplifiedStmt;
}

}}
