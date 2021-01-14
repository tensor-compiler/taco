#ifndef TACO_LINALG_NOTATION_PRINTER_H
#define TACO_LINALG_NOTATION_PRINTER_H

#include <ostream>
#include "taco/linalg_notation/linalg_notation_visitor.h"

namespace taco {

class LinalgNotationPrinter : public LinalgNotationVisitorStrict {
public:
  explicit LinalgNotationPrinter(std::ostream& os);

  void print(const LinalgExpr& expr);
  void print(const LinalgStmt& expr);

  using LinalgExprVisitorStrict::visit;

  // Scalar Expressions
  void visit(const LinalgVarNode*);
  void visit(const LinalgTensorBaseNode*);
  void visit(const LinalgLiteralNode*);
  void visit(const LinalgNegNode*);
  void visit(const LinalgAddNode*);
  void visit(const LinalgSubNode*);
  void visit(const LinalgMatMulNode*);
  void visit(const LinalgElemMulNode*);
  void visit(const LinalgDivNode*);
  void visit(const LinalgTransposeNode*);

  void visit(const LinalgAssignmentNode*);

private:
  std::ostream& os;

  enum class Precedence {
    ACCESS = 2,
    VAR = 2,
    FUNC = 2,
    NEG = 3,
    TRANSPOSE = 3,
    MATMUL = 5,
    ELEMMUL = 5,
    DIV = 5,
    ADD = 6,
    SUB = 6,
    TOP = 20
  };
  Precedence parentPrecedence;

  template <typename Node> void visitBinary(Node op, Precedence p);
};

}
#endif //TACO_LINALG_NOTATION_PRINTER_H
