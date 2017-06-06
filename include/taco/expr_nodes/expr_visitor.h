#ifndef TACO_EXPR_VISITOR_H
#define TACO_EXPR_VISITOR_H

#include <functional>
#include "taco/error.h"

namespace taco {
class IndexExpr;

namespace expr_nodes {

struct ReadNode;
struct NegNode;
struct SqrtNode;
struct AddNode;
struct SubNode;
struct MulNode;
struct DivNode;
struct IntImmNode;
struct FloatImmNode;
struct DoubleImmNode;
struct ImmExprNode;
struct UnaryExprNode;
struct BinaryExprNode;

/// Visit the nodes in an expression.  This visitor provides some type safety
/// by requing all visit methods to be overridden.
class ExprVisitorStrict {
public:
  virtual ~ExprVisitorStrict();

  void visit(const IndexExpr& expr);

  virtual void visit(const ReadNode*) = 0;
  virtual void visit(const NegNode*) = 0;
  virtual void visit(const SqrtNode*) = 0;
  virtual void visit(const AddNode*) = 0;
  virtual void visit(const SubNode*) = 0;
  virtual void visit(const MulNode*) = 0;
  virtual void visit(const DivNode*) = 0;
  virtual void visit(const IntImmNode*) = 0;
  virtual void visit(const FloatImmNode*) = 0;
  virtual void visit(const DoubleImmNode*) = 0;
};


/// Visit nodes in an expression.
class ExprVisitor : public ExprVisitorStrict {
public:
  using ExprVisitorStrict::visit;

  virtual ~ExprVisitor();

  virtual void visit(const ReadNode* op);
  virtual void visit(const NegNode* op);
  virtual void visit(const SqrtNode* op);
  virtual void visit(const AddNode* op);
  virtual void visit(const SubNode* op);
  virtual void visit(const MulNode* op);
  virtual void visit(const DivNode* op);
  virtual void visit(const IntImmNode* op);
  virtual void visit(const FloatImmNode* op);
  virtual void visit(const DoubleImmNode* op);

  virtual void visit(const ImmExprNode*);
  virtual void visit(const UnaryExprNode*);
  virtual void visit(const BinaryExprNode*);
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
 ExprVisitor::visit(op);                                                       \
}

class Matcher : public ExprVisitor {
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

  using ExprVisitor::visit;
  RULE(ReadNode)
  RULE(NegNode)
  RULE(SqrtNode)
  RULE(AddNode)
  RULE(SubNode)
  RULE(MulNode)
  RULE(DivNode)
  RULE(IntImmNode)
  RULE(FloatImmNode)
  RULE(DoubleImmNode)
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
    std::function<void(const SubNode*,Matcher* ctx)>([&](const SubNode* op
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


}}
#endif
