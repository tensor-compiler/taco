#include "taco/index_notation/index_notation_rewriter.h"

#include "taco/index_notation/index_notation_nodes.h"
#include "taco/util/collections.h"

#include <vector>

using namespace std;

namespace taco {

// class ExprRewriterStrict
IndexExpr IndexExprRewriterStrict::rewrite(IndexExpr e) {
  if (e.defined()) {
    e.accept(this);
    e = expr;
  }
  else {
    e = IndexExpr();
  }
  expr = IndexExpr();
  return e;
}


// class IndexStmtRewriterStrict
IndexStmt IndexStmtRewriterStrict::rewrite(IndexStmt s) {
  if (s.defined()) {
    s.accept(this);
    s = stmt;
  }
  else {
    s = IndexStmt();
  }
  stmt = IndexStmt();
  return s;
}


// class ExprRewriter
void IndexNotationRewriter::visit(const AccessNode* op) {
  expr = op;
}

void IndexNotationRewriter::visit(const IndexVarNode* op) {
  expr = op;
}

template <class T>
IndexExpr visitUnaryOp(const T *op, IndexNotationRewriter *rw) {
  IndexExpr a = rw->rewrite(op->a);
  if (a == op->a) {
    return op;
  }
  else {
    return new T(a);
  }
}

void IndexNotationRewriter::visit(const LiteralNode* op) {
  expr = op;
}

template <class T>
IndexExpr visitBinaryOp(const T *op, IndexNotationRewriter *rw) {
  IndexExpr a = rw->rewrite(op->a);
  IndexExpr b = rw->rewrite(op->b);
  if (a == op->a && b == op->b) {
    return op;
  }
  else {
    return new T(a, b);
  }
}

void IndexNotationRewriter::visit(const NegNode* op) {
  expr = visitUnaryOp(op, this);
}

void IndexNotationRewriter::visit(const SqrtNode* op) {
  expr = visitUnaryOp(op, this);
}

void IndexNotationRewriter::visit(const AddNode* op) {
  expr = visitBinaryOp(op, this);
}

void IndexNotationRewriter::visit(const SubNode* op) {
  expr = visitBinaryOp(op, this);
}

void IndexNotationRewriter::visit(const MulNode* op) {
  expr = visitBinaryOp(op, this);
}

void IndexNotationRewriter::visit(const DivNode* op) {
  expr = visitBinaryOp(op, this);
}

void IndexNotationRewriter::visit(const CastNode* op) {
  IndexExpr a = rewrite(op->a);
  if (a == op->a) {
    expr = op;
  }
  else {
    expr = new CastNode(a, op->getDataType());
  }
}

void IndexNotationRewriter::visit(const CallNode* op) {
  std::vector<IndexExpr> args;
  bool rewritten = false;
  for(auto& arg : op->args) {
    IndexExpr rewrittenArg = rewrite(arg);
    args.push_back(rewrittenArg);
    if (arg != rewrittenArg) {
      rewritten = true;
    }
  }

  if (rewritten) {
    const std::map<IndexExpr, IndexExpr> subs = util::zipToMap(op->args, args);
    IterationAlgebra newAlg = replaceAlgIndexExprs(op->iterAlg, subs);
    expr = new CallNode(op->name, args, op->defaultLowerFunc, newAlg, op->properties,
                        op->regionDefinitions);
  }
  else {
    expr = op;
  }
}

void IndexNotationRewriter::visit(const CallIntrinsicNode* op) {
  std::vector<IndexExpr> args;
  bool rewritten = false;
  for (auto& arg : op->args) {
    IndexExpr rewrittenArg = rewrite(arg);
    args.push_back(rewrittenArg);
    if (arg != rewrittenArg) {
      rewritten = true;
    }
  }
  if (rewritten) {
    expr = new CallIntrinsicNode(op->func, args);
  }
  else {
    expr = op;
  }
}

void IndexNotationRewriter::visit(const ReductionNode* op) {
  IndexExpr a = rewrite(op->a);
  if (a == op->a) {
    expr = op;
  }
  else {
    expr = new ReductionNode(op->op, op->var, a);
  }
}

void IndexNotationRewriter::visit(const AssignmentNode* op) {
  // A design decission is to not visit the lhs access expressions or the op,
  // as these are considered part of the assignment.  When visiting access
  // expressions, therefore, we only visit read access expressions.
  IndexExpr rhs = rewrite(op->rhs);
  if (rhs == op->rhs) {
    stmt = op;
  }
  else {
    stmt = new AssignmentNode(op->lhs, rhs, op->op);
  }
}

void IndexNotationRewriter::visit(const YieldNode* op) {
  IndexExpr expr = rewrite(op->expr);
  if (expr == op->expr) {
    stmt = op;
  } else {
    stmt = new YieldNode(op->indexVars, expr);
  }
}

void IndexNotationRewriter::visit(const ForallNode* op) {
  IndexStmt s = rewrite(op->stmt);
  if (s == op->stmt) {
    stmt = op;
  }
  else {
    stmt = new ForallNode(op->indexVar, s, op->merge_strategy, op->parallel_unit, op->output_race_strategy, op->unrollFactor);
  }
}

void IndexNotationRewriter::visit(const WhereNode* op) {
  IndexStmt producer = rewrite(op->producer);
  IndexStmt consumer = rewrite(op->consumer);
  if (producer == op->producer && consumer == op->consumer) {
    stmt = op;
  }
  else {
    stmt = new WhereNode(consumer, producer);
  }
}

void IndexNotationRewriter::visit(const SequenceNode* op) {
  IndexStmt definition = rewrite(op->definition);
  IndexStmt mutation = rewrite(op->mutation);
  if (definition == op->definition && mutation == op->mutation) {
    stmt = op;
  }
  else {
    stmt = new SequenceNode(definition, mutation);
  }
}

void IndexNotationRewriter::visit(const AssembleNode* op) {
  IndexStmt queries = rewrite(op->queries);
  IndexStmt compute = rewrite(op->compute);
  if (queries == op->queries && compute == op->compute) {
    stmt = op;
  }
  else {
    stmt = new AssembleNode(queries, compute, op->results);
  }
}

void IndexNotationRewriter::visit(const MultiNode* op) {
  IndexStmt stmt1 = rewrite(op->stmt1);
  IndexStmt stmt2 = rewrite(op->stmt2);
  if (stmt1 == op->stmt1 && stmt2 == op->stmt2) {
    stmt = op;
  }
  else {
    stmt = new MultiNode(stmt1, stmt2);
  }
}

void IndexNotationRewriter::visit(const SuchThatNode* op) {
  IndexStmt s = rewrite(op->stmt);
  if (s == op->stmt) {
    stmt = op;
  }
  else {
    stmt = new SuchThatNode(s, op->predicate);
  }
}



// Functions
#define SUBSTITUTE_EXPR                        \
do {                                           \
  IndexExpr e = op;                            \
  if (util::contains(exprSubstitutions, e)) {  \
    expr = exprSubstitutions.at(e);            \
  }                                            \
  else {                                       \
    IndexNotationRewriter::visit(op);          \
  }                                            \
} while(false)

#define SUBSTITUTE_STMT                        \
do {                                           \
  IndexStmt s = op;                            \
  if (util::contains(stmtSubstitutions, s)) {  \
    stmt = stmtSubstitutions.at(s);            \
  }                                            \
  else {                                       \
    IndexNotationRewriter::visit(op);          \
  }                                            \
} while(false)

struct ReplaceRewriter : public IndexNotationRewriter {
  const std::map<IndexExpr,IndexExpr>& exprSubstitutions;
  const std::map<IndexStmt,IndexStmt>& stmtSubstitutions;

  ReplaceRewriter(const std::map<IndexExpr,IndexExpr>& exprSubstitutions,
                  const std::map<IndexStmt,IndexStmt>& stmtSubstitutions)
      : exprSubstitutions(exprSubstitutions),
        stmtSubstitutions(stmtSubstitutions) {}

  using IndexNotationRewriter::visit;

  void visit(const AccessNode* op) {
    SUBSTITUTE_EXPR;
  }

  void visit(const IndexVarNode* op) {
    SUBSTITUTE_EXPR;
  }

  void visit(const LiteralNode* op) {
    SUBSTITUTE_EXPR;
  }

  void visit(const NegNode* op) {
    SUBSTITUTE_EXPR;
  }

  void visit(const SqrtNode* op) {
    SUBSTITUTE_EXPR;
  }

  void visit(const AddNode* op) {
    SUBSTITUTE_EXPR;
  }

  void visit(const SubNode* op) {
    SUBSTITUTE_EXPR;
  }

  void visit(const MulNode* op) {
    SUBSTITUTE_EXPR;
  }

  void visit(const DivNode* op) {
    SUBSTITUTE_EXPR;
  }

  void visit(const CallNode* op) {
    SUBSTITUTE_EXPR;
  }

  void visit(const CallIntrinsicNode* op) {
    SUBSTITUTE_EXPR;
  }

  void visit(const ReductionNode* op) {
    SUBSTITUTE_EXPR;
  }

  void visit(const AssignmentNode* op) {
    SUBSTITUTE_STMT;
  }

  void visit(const YieldNode* op) {
    SUBSTITUTE_STMT;
  }

  void visit(const ForallNode* op) {
    SUBSTITUTE_STMT;
  }

  void visit(const WhereNode* op) {
    SUBSTITUTE_STMT;
  }

  void visit(const SequenceNode* op) {
    SUBSTITUTE_STMT;
  }

  void visit(const AssembleNode* op) {
    SUBSTITUTE_STMT;
  }

  void visit(const MultiNode* op) {
    SUBSTITUTE_STMT;
  }

  void visit(const SuchThatNode* op) {
    SUBSTITUTE_STMT;
  }
};

struct ReplaceIndexVars : public IndexNotationRewriter {
  const std::map<IndexVar,IndexVar>& substitutions;
  ReplaceIndexVars(const std::map<IndexVar,IndexVar>& substitutions)
      : substitutions(substitutions) {}

  using IndexNotationRewriter::visit;

  void visit(const AccessNode* op) {
    vector<IndexVar> indexVars;
    bool modified = false;
    for (auto& var : op->indexVars) {
      if (util::contains(substitutions, var)) {
        indexVars.push_back(substitutions.at(var));
        modified = true;
      }
      else {
        indexVars.push_back(var);
      }
    }
    if (modified) {
      expr = Access(op->tensorVar, indexVars, op->packageModifiers());
    }
    else {
      expr = op;
    }
  }

  void visit(const AssignmentNode* op) {
    IndexExpr rhs = rewrite(op->rhs);
    Access lhs = to<Access>(rewrite(op->lhs));
    if (rhs == op->rhs && lhs == op->lhs) {
      stmt = op;
    }
    else {
      stmt = new AssignmentNode(lhs, rhs, op->op);
    }
  }

  void visit(const ForallNode* op) {
    IndexStmt s = rewrite(op->stmt);
    IndexVar iv = util::contains(substitutions, op->indexVar) 
                ? substitutions.at(op->indexVar) : op->indexVar;
    if (s == op->stmt && iv == op->indexVar) {
      stmt = op;
    }
    else {
      stmt = new ForallNode(iv, s, op->merge_strategy, op->parallel_unit, op->output_race_strategy, 
                            op->unrollFactor);
    }
  }

  void visit(const IndexVarNode* op) {
    IndexVar var(op);
    if(util::contains(substitutions, var)) {
      expr = substitutions.at(var);
    } else {
      expr = var;
    }
  }
};

struct ReplaceTensorVars : public IndexNotationRewriter {
  const std::map<TensorVar,TensorVar>& substitutions;
  ReplaceTensorVars(const std::map<TensorVar,TensorVar>& substitutions)
      : substitutions(substitutions) {}

  using IndexNotationRewriter::visit;

  void visit(const AccessNode* op) {
    TensorVar var = op->tensorVar;
    expr = (util::contains(substitutions, var))
           ? Access(substitutions.at(var), op->indexVars)
           : op;
  }

  void visit(const AssignmentNode* node) {
    TensorVar var = node->lhs.getTensorVar();
    if (util::contains(substitutions, var)) {
      stmt = Assignment(substitutions.at(var),
                        node->lhs.getIndexVars(), rewrite(node->rhs),
                        node->op);
    }
    else {
      IndexNotationRewriter::visit(node);
    }
  }
};

IndexExpr replace(IndexExpr expr,
                  const std::map<IndexExpr,IndexExpr>& substitutions) {
  return ReplaceRewriter(substitutions, {}).rewrite(expr);
}

IndexExpr replace(IndexExpr expr,
                  const std::map<IndexVar,IndexVar>& substitutions) {
  return ReplaceIndexVars(substitutions).rewrite(expr);
}

IndexStmt replace(IndexStmt stmt,
                  const std::map<IndexExpr,IndexExpr>& substitutions) {
  return ReplaceRewriter(substitutions,{}).rewrite(stmt);
}

IndexStmt replace(IndexStmt stmt,
                  const std::map<IndexStmt,IndexStmt>& substitutions) {
  return ReplaceRewriter({}, substitutions).rewrite(stmt);
}

IndexStmt replace(IndexStmt stmt,
                  const std::map<TensorVar,TensorVar>& substitutions) {
  return ReplaceTensorVars(substitutions).rewrite(stmt);
}

IndexStmt replace(IndexStmt stmt,
                  const std::map<IndexVar,IndexVar>& substitutions) {
  return ReplaceIndexVars(substitutions).rewrite(stmt);
}

}
