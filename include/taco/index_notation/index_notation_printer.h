#ifndef TACO_INDEX_NOTATION_PRINTER_H
#define TACO_INDEX_NOTATION_PRINTER_H

#include <ostream>
#include "taco/index_notation/index_notation_visitor.h"

namespace taco {

class IndexNotationPrinter : public IndexNotationVisitorStrict {
public:
  IndexNotationPrinter(std::ostream& os);

  void print(const IndexExpr& expr);
  void print(const IndexStmt& expr);

  using IndexNotationVisitorStrict::visit;

  // Scalar Expressions
  void visit(const AccessNode*);
  void visit(const LiteralNode*);
  void visit(const NegNode*);
  void visit(const SqrtNode*);
  void visit(const AddNode*);
  void visit(const SubNode*);
  void visit(const MulNode*);
  void visit(const DivNode*);
  void visit(const CastNode*);
  void visit(const CallIntrinsicNode*);
  void visit(const ReductionNode*);

  // Tensor Expressions
  void visit(const AssignmentNode*);
  void visit(const YieldNode*);
  void visit(const ForallNode*);
  void visit(const WhereNode*);
  void visit(const MultiNode*);
  void visit(const SequenceNode*);

private:
  std::ostream& os;

  enum class Precedence {
    ACCESS = 2,
    FUNC = 2,
    CAST = 2,
    REDUCTION = 2,
    NEG = 3,
    MUL = 5,
    DIV = 5,
    ADD = 6,
    SUB = 6,
    TOP = 20
  };
  Precedence parentPrecedence;

  template <typename Node> void visitBinary(Node op, Precedence p);
  template <typename Node> void visitImmediate(Node op);
};

}
#endif
