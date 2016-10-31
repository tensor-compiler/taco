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

class ExprVisitor {
public:
  virtual ~ExprVisitor();
  virtual void visit(const Read*);
  virtual void visit(const Neg*);
  virtual void visit(const Sqrt*);
  virtual void visit(const Add*);
  virtual void visit(const Sub*);
  virtual void visit(const Mul*);
  virtual void visit(const Div*);
  virtual void visit(const IntImm*);
  virtual void visit(const FloatImm*);
  virtual void visit(const DoubleImm*);
};

}}
#endif
