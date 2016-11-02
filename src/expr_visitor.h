#ifndef TACO_EXPR_VISITOR_H
#define TACO_EXPR_VISITOR_H

namespace taco {
namespace internal {

struct Read;
struct Neg;
struct Sqrt;
struct Add;
struct Sub;
struct Mul;
struct Div;

struct IntImm;
struct FloatImm;
struct DoubleImm;

class ExprVisitorStrict {
public:
  virtual ~ExprVisitorStrict();
  virtual void visit(const Read*) = 0;
  virtual void visit(const Neg*) = 0;
  virtual void visit(const Sqrt*) = 0;
  virtual void visit(const Add*) = 0;
  virtual void visit(const Sub*) = 0;
  virtual void visit(const Mul*) = 0;
  virtual void visit(const Div*) = 0;
  virtual void visit(const IntImm*) = 0;
  virtual void visit(const FloatImm*) = 0;
  virtual void visit(const DoubleImm*) = 0;
};

class ExprVisitor : public ExprVisitorStrict {
public:
  virtual ~ExprVisitor();
  virtual void visit(const Read* op);
  virtual void visit(const Neg* op);
  virtual void visit(const Sqrt* op);
  virtual void visit(const Add* op);
  virtual void visit(const Sub* op);
  virtual void visit(const Mul* op);
  virtual void visit(const Div* op);
  virtual void visit(const IntImm* op);
  virtual void visit(const FloatImm* op);
  virtual void visit(const DoubleImm* op);
};

}}
#endif
