#include "taco/index_notation/iteration_algebra_printer.h"

namespace taco {

// Iteration Algebra Printer
IterationAlgebraPrinter::IterationAlgebraPrinter(std::ostream& os) : os(os) {}

void IterationAlgebraPrinter::print(const IterationAlgebra& alg) {
  parentPrecedence = Precedence::TOP;
  alg.accept(this);
}

void IterationAlgebraPrinter::visit(const RegionNode* n) {
  os << n->expr();
}

void IterationAlgebraPrinter::visit(const ComplementNode* n) {
  Precedence precedence = Precedence::COMPLEMENT;
  bool parenthesize =  precedence > parentPrecedence;
  parentPrecedence = precedence;
  os << "~";
  if (parenthesize) {
    os << "(";
  }
  n->a.accept(this);
  if (parenthesize) {
    os << ")";
  }
}

void IterationAlgebraPrinter::visit(const IntersectNode* n) {
  visitBinary(n, Precedence::INTERSECT);
}

void IterationAlgebraPrinter::visit(const UnionNode* n) {
  visitBinary(n, Precedence::UNION);
}

template <typename Node>
void IterationAlgebraPrinter::visitBinary(Node n, Precedence precedence) {
  bool parenthesize =  precedence > parentPrecedence;
  if (parenthesize) {
    os << "(";
  }
  parentPrecedence = precedence;
  n->a.accept(this);
  os << " " << n->algebraString() << " ";
  parentPrecedence = precedence;
  n->b.accept(this);
  if (parenthesize) {
    os << ")";
  }
}

}