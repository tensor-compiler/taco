#ifndef TACO_EXPR_VISITOR_H
#define TACO_EXPR_VISITOR_H

namespace taco {
class Expr;

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

class ExprVisitorStrict {
public:
  virtual ~ExprVisitorStrict();

  void visit(const Expr& expr);

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

}}
#endif
