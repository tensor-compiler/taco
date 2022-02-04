#ifndef TACO_ITERATION_ALGEBRA_PRINTER_H
#define TACO_ITERATION_ALGEBRA_PRINTER_H

#include <ostream>
#include "taco/index_notation/iteration_algebra.h"

namespace taco {

// Iteration Algebra Printer
class IterationAlgebraPrinter : IterationAlgebraVisitorStrict {
public:
  IterationAlgebraPrinter(std::ostream& os);
  void print(const IterationAlgebra& alg);
  void visit(const RegionNode* n);
  void visit(const ComplementNode* n);
  void visit(const IntersectNode* n);
  void visit(const UnionNode* n);

private:
  std::ostream& os;
  enum class Precedence {
    COMPLEMENT = 3,
    INTERSECT = 4,
    UNION = 5,
    TOP = 20
  };

  Precedence parentPrecedence;

  template <typename Node>
  void visitBinary(Node n, Precedence precedence);
};
}
#endif //TACO_ITERATION_ALGEBRA_PRINTER_H
