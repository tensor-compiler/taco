#ifndef TACO_LINALG_NOTATION_VISITOR_H
#define TACO_LINALG_NOTATION_VISITOR_H
namespace taco {

class LinalgExpr;
class LinalgStmt;

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

struct LinalgAssignmentNode;

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

class LinalgStmtVisitorStrict {
public:
  virtual ~LinalgStmtVisitorStrict() = default;

  void visit(const LinalgStmt&);

  virtual void visit(const LinalgAssignmentNode*) = 0;
};

/// Visit nodes in linalg notation
class LinalgNotationVisitorStrict : public LinalgExprVisitorStrict,
                                   public LinalgStmtVisitorStrict {
public:
  virtual ~LinalgNotationVisitorStrict() = default;

  using LinalgExprVisitorStrict::visit;
  using LinalgStmtVisitorStrict::visit;
};

/// Visit nodes in an expression.
class LinalgNotationVisitor : public LinalgNotationVisitorStrict {
public:
  virtual ~LinalgNotationVisitor() = default;

  using LinalgNotationVisitorStrict::visit;

  // Index Expressions
  virtual void visit(const VarNode* node);
  virtual void visit(const LiteralNode* node);
  virtual void visit(const NegNode* node);
  virtual void visit(const AddNode* node);
  virtual void visit(const SubNode* node);
  virtual void visit(const MatMulNode* node);
  virtual void visit(const ElemMulNode* node);
  virtual void visit(const DivNode* node);
  virtual void visit(const UnaryExprNode* node);
  virtual void visit(const BinaryExprNode* node);
  virtual void visit(const TransposeNode* node);

  // Index Statments
  virtual void visit(const LinalgAssignmentNode* node);
};

}
#endif //TACO_LINALG_NOTATION_VISITOR_H
