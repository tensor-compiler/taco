#ifndef TACO_EXPR_VISITOR_H
#define TACO_EXPR_VISITOR_H

namespace taco {

struct IntImmNode;
struct FloatImmNode;
struct DoubleImmNode;
struct ReadNode;
struct AddNode;
struct SubNode;
struct MulNode;
struct DivNode;

namespace internal {

class ExprVisitor {
public:
  virtual ~ExprVisitor();
  virtual void visit(const IntImmNode*);
  virtual void visit(const FloatImmNode*);
  virtual void visit(const DoubleImmNode*);
  virtual void visit(const ReadNode*);
  virtual void visit(const AddNode*);
  virtual void visit(const SubNode*);
  virtual void visit(const MulNode*);
  virtual void visit(const DivNode*);
};

}}
#endif
