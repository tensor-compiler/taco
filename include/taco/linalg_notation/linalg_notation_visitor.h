#ifndef TACO_LINALG_NOTATION_VISITOR_H
#define TACO_LINALG_NOTATION_VISITOR_H
namespace taco {

class LinalgExpr;
class LinalgStmt;

class TensorVar;

struct LinalgVarNode;
struct LinalgTensorBaseNode;
struct LinalgLiteralNode;
struct LinalgNegNode;
struct LinalgTransposeNode;
struct LinalgAddNode;
struct LinalgSubNode;
struct LinalgMatMulNode;
struct LinalgElemMulNode;
struct LinalgDivNode;
struct LinalgUnaryExprNode;
struct LinalgBinaryExprNode;

struct LinalgAssignmentNode;

/// Visit the nodes in an expression.  This visitor provides some type safety
/// by requiring all visit methods to be overridden.
class LinalgExprVisitorStrict {
public:
  virtual ~LinalgExprVisitorStrict() = default;

  void visit(const LinalgExpr &);

  virtual void visit(const LinalgVarNode *) = 0;

  virtual void visit(const LinalgTensorBaseNode*) = 0;

  virtual void visit(const LinalgLiteralNode *) = 0;

  virtual void visit(const LinalgNegNode *) = 0;

  virtual void visit(const LinalgAddNode *) = 0;

  virtual void visit(const LinalgSubNode *) = 0;

  virtual void visit(const LinalgMatMulNode *) = 0;

  virtual void visit(const LinalgElemMulNode *) = 0;

  virtual void visit(const LinalgDivNode *) = 0;

  virtual void visit(const LinalgTransposeNode *) = 0;
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
  virtual void visit(const LinalgVarNode* node);
  virtual void visit(const LinalgTensorBaseNode* node);
  virtual void visit(const LinalgLiteralNode* node);
  virtual void visit(const LinalgNegNode* node);
  virtual void visit(const LinalgAddNode* node);
  virtual void visit(const LinalgSubNode* node);
  virtual void visit(const LinalgMatMulNode* node);
  virtual void visit(const LinalgElemMulNode* node);
  virtual void visit(const LinalgDivNode* node);
  virtual void visit(const LinalgUnaryExprNode* node);
  virtual void visit(const LinalgBinaryExprNode* node);
  virtual void visit(const LinalgTransposeNode* node);

  // Index Statments
  virtual void visit(const LinalgAssignmentNode* node);
};

}
#endif //TACO_LINALG_NOTATION_VISITOR_H
