#ifndef TACO_LINALG_NOTATION_VISITOR_H
#define TACO_LINALG_NOTATION_VISITOR_H
namespace taco {

class LinalgExpr;

class TensorVar;

struct VarNode;
struct LiteralNode;
struct NegNode;
struct TransposeNode;
struct AddNode;
struct SubNode;
struct MatMulNode;
struct ElemMulNode;
struct DivNode;
struct UnaryExprNode;
struct BinaryExprNode;



/// Visit the nodes in an expression.  This visitor provides some type safety
/// by requiring all visit methods to be overridden.
class LinalgExprVisitorStrict {
public:
  virtual ~LinalgExprVisitorStrict() = default;

  void visit(const LinalgExpr &);

  virtual void visit(const VarNode *) = 0;

  virtual void visit(const LiteralNode *) = 0;

  virtual void visit(const NegNode *) = 0;

  virtual void visit(const AddNode *) = 0;

  virtual void visit(const SubNode *) = 0;

  virtual void visit(const MatMulNode *) = 0;

  virtual void visit(const ElemMulNode *) = 0;

  virtual void visit(const DivNode *) = 0;

  virtual void visit(const TransposeNode *) = 0;
};

}
#endif //TACO_LINALG_NOTATION_VISITOR_H
