#ifndef TACO_INDEX_NOTATION_VISITOR_H
#define TACO_INDEX_NOTATION_VISITOR_H

#include <vector>
#include <functional>
#include "taco/error.h"

namespace taco {

class IndexStmt;
class IndexExpr;
class TensorVar;

struct AccessNode;
struct LiteralNode;
struct NegNode;
struct AddNode;
struct SubNode;
struct MulNode;
struct DivNode;
struct SqrtNode;
struct CastNode;
struct CallIntrinsicNode;
struct UnaryExprNode;
struct BinaryExprNode;
struct ReductionNode;

struct AssignmentNode;
struct YieldNode;
struct ForallNode;
struct WhereNode;
struct MultiNode;
struct SequenceNode;

/// Visit the nodes in an expression.  This visitor provides some type safety
/// by requing all visit methods to be overridden.
class IndexExprVisitorStrict {
public:
  virtual ~IndexExprVisitorStrict();

  void visit(const IndexExpr&);

  virtual void visit(const AccessNode*) = 0;
  virtual void visit(const LiteralNode*) = 0;
  virtual void visit(const NegNode*) = 0;
  virtual void visit(const AddNode*) = 0;
  virtual void visit(const SubNode*) = 0;
  virtual void visit(const MulNode*) = 0;
  virtual void visit(const DivNode*) = 0;
  virtual void visit(const SqrtNode*) = 0;
  virtual void visit(const CastNode*) = 0;
  virtual void visit(const CallIntrinsicNode*) = 0;
  virtual void visit(const ReductionNode*) = 0;
};

class IndexStmtVisitorStrict {
public:
  virtual ~IndexStmtVisitorStrict();

  void visit(const IndexStmt&);

  virtual void visit(const AssignmentNode*) = 0;
  virtual void visit(const YieldNode*) = 0;
  virtual void visit(const ForallNode*) = 0;
  virtual void visit(const WhereNode*) = 0;
  virtual void visit(const SequenceNode*) = 0;
  virtual void visit(const MultiNode*) = 0;
};

/// Visit nodes in index notation
class IndexNotationVisitorStrict : public IndexExprVisitorStrict,
                                   public IndexStmtVisitorStrict {
public:
  virtual ~IndexNotationVisitorStrict();

  using IndexExprVisitorStrict::visit;
  using IndexStmtVisitorStrict::visit;
};

/// Visit nodes in an expression.
class IndexNotationVisitor : public IndexNotationVisitorStrict {
public:
  virtual ~IndexNotationVisitor();

  using IndexNotationVisitorStrict::visit;

  // Index Expressions
  virtual void visit(const AccessNode* node);
  virtual void visit(const LiteralNode* node);
  virtual void visit(const NegNode* node);
  virtual void visit(const AddNode* node);
  virtual void visit(const SubNode* node);
  virtual void visit(const MulNode* node);
  virtual void visit(const DivNode* node);
  virtual void visit(const SqrtNode* node);
  virtual void visit(const CastNode* node);
  virtual void visit(const CallIntrinsicNode* node);
  virtual void visit(const UnaryExprNode* node);
  virtual void visit(const BinaryExprNode* node);
  virtual void visit(const ReductionNode* node);

  // Index Statments
  virtual void visit(const AssignmentNode* node);
  virtual void visit(const YieldNode* node);
  virtual void visit(const ForallNode* node);
  virtual void visit(const WhereNode* node);
  virtual void visit(const SequenceNode* node);
  virtual void visit(const MultiNode* node);
};


#define RULE(Rule)                                                             \
std::function<void(const Rule*)> Rule##Func;                                   \
std::function<void(const Rule*, Matcher*)> Rule##CtxFunc;                      \
void unpack(std::function<void(const Rule*)> pattern) {                        \
  taco_iassert(!Rule##CtxFunc && !Rule##Func);                                 \
  Rule##Func = pattern;                                                        \
}                                                                              \
void unpack(std::function<void(const Rule*, Matcher*)> pattern) {              \
  taco_iassert(!Rule##CtxFunc && !Rule##Func);                                 \
  Rule##CtxFunc = pattern;                                                     \
}                                                                              \
void visit(const Rule* op) {                                                   \
  if (Rule##Func) {                                                            \
    Rule##Func(op);                                                            \
  }                                                                            \
  else if (Rule##CtxFunc) {                                                    \
    Rule##CtxFunc(op, this);                                                   \
    return;                                                                    \
  }                                                                            \
 IndexNotationVisitor::visit(op);                                              \
}

class Matcher : public IndexNotationVisitor {
public:
  template <class IndexExpr>
  void match(IndexExpr indexExpr) {
    indexExpr.accept(this);
  }

  template <class IR, class... Patterns>
  void process(IR ir, Patterns... patterns) {
    unpack(patterns...);
    ir.accept(this);
  }

private:
  template <class First, class... Rest>
  void unpack(First first, Rest... rest) {
    unpack(first);
    unpack(rest...);
  }

  using IndexNotationVisitor::visit;
  RULE(AccessNode)
  RULE(LiteralNode)
  RULE(NegNode)
  RULE(SqrtNode)
  RULE(AddNode)
  RULE(SubNode)
  RULE(MulNode)
  RULE(DivNode)
  RULE(CastNode)
  RULE(CallIntrinsicNode)
  RULE(ReductionNode)

  RULE(BinaryExprNode)
  RULE(UnaryExprNode)

  RULE(AssignmentNode)
  RULE(YieldNode)
  RULE(ForallNode)
  RULE(WhereNode)
  RULE(MultiNode)
  RULE(SequenceNode)
};

/**
  Match patterns to the IR.  Use lambda closures to capture environment
  variables (e.g. [&]):

  For example, to print all AddNode and SubNode objects in func:
  ~~~~~~~~~~~~~~~{.cpp}
  match(expr,
    std::function<void(const AddNode*)>([](const AddNode* op) {
      // ...
    })
    ,
    std::function<void(const SubNode*)>([](const SubNode* op) {
      // ...
    })
  );
  ~~~~~~~~~~~~~~~

  Alternatively, mathing rules can also accept a Context to be used to match
  sub-expressions:
  ~~~~~~~~~~~~~~~{.cpp}
  match(expr,
    std::function<void(const SubNode*,Matcher*)>([&](const SubNode* op,
                                                     Matcher* ctx){
      ctx->match(op->a);
    })
  );
  ~~~~~~~~~~~~~~~

  function<void(const Add*, Matcher* ctx)>([&](const Add* op, Matcher* ctx) {
**/
template <class IndexExpr, class... Patterns>
void match(IndexExpr indexExpr, Patterns... patterns) {
  if (!indexExpr.defined()) {
    return;
  }
  Matcher().process(indexExpr, patterns...);
}

}
#endif
