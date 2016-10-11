#ifndef TACO_EXPR_VISITOR_H
#define TACO_EXPR_VISITOR_H

namespace taco {
namespace internal {

struct ReadNode;
struct NegNode;
struct AddNode;
struct SubNode;
struct MulNode;
struct DivNode;

struct IntImmNode;
struct FloatImmNode;
struct DoubleImmNode;

class ExprVisitor {
public:
  virtual ~ExprVisitor();
  virtual void visit(const ReadNode*);
  virtual void visit(const NegNode*);
  virtual void visit(const AddNode*);
  virtual void visit(const SubNode*);
  virtual void visit(const MulNode*);
  virtual void visit(const DivNode*);
  virtual void visit(const IntImmNode*);
  virtual void visit(const FloatImmNode*);
  virtual void visit(const DoubleImmNode*);
};

}}
#endif
