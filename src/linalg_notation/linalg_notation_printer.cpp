#include "taco/linalg_notation/linalg_notation_printer.h"
#include "taco/linalg_notation/linalg_notation_nodes.h"

using namespace std;

namespace taco {

LinalgNotationPrinter::LinalgNotationPrinter(std::ostream& os) : os(os) {
}

void LinalgNotationPrinter::print(const LinalgExpr& expr) {
  parentPrecedence = Precedence::TOP;
  expr.accept(this);
}

void LinalgNotationPrinter::print(const LinalgStmt& expr) {
  parentPrecedence = Precedence::TOP;
  expr.accept(this);
}

void LinalgNotationPrinter::visit(const LinalgVarNode* op) {
  os << op->tensorVar.getName();
}

void LinalgNotationPrinter::visit(const LinalgTensorBaseNode* op) {
  os << op->tensorBase->getName();
}

void LinalgNotationPrinter::visit(const LinalgLiteralNode* op) {
  switch (op->getDataType().getKind()) {
    case Datatype::Bool:
      os << op->getVal<bool>();
      break;
    case Datatype::UInt8:
      os << op->getVal<uint8_t>();
      break;
    case Datatype::UInt16:
      os << op->getVal<uint16_t>();
      break;
    case Datatype::UInt32:
      os << op->getVal<uint32_t>();
      break;
    case Datatype::UInt64:
      os << op->getVal<uint64_t>();
      break;
    case Datatype::UInt128:
      taco_not_supported_yet;
      break;
    case Datatype::Int8:
      os << op->getVal<int8_t>();
      break;
    case Datatype::Int16:
      os << op->getVal<int16_t>();
      break;
    case Datatype::Int32:
      os << op->getVal<int32_t>();
      break;
    case Datatype::Int64:
      os << op->getVal<int64_t>();
      break;
    case Datatype::Int128:
      taco_not_supported_yet;
      break;
    case Datatype::Float32:
      os << op->getVal<float>();
      break;
    case Datatype::Float64:
      os << op->getVal<double>();
      break;
    case Datatype::Complex64:
      os << op->getVal<std::complex<float>>();
      break;
    case Datatype::Complex128:
      os << op->getVal<std::complex<double>>();
      break;
    case Datatype::Undefined:
      break;
  }
}

void LinalgNotationPrinter::visit(const LinalgNegNode* op) {
  Precedence precedence = Precedence::NEG;
  bool parenthesize =  precedence > parentPrecedence;
  parentPrecedence = precedence;
  os << "-";
  if (parenthesize) {
    os << "(";
  }
  op->a.accept(this);
  if (parenthesize) {
    os << ")";
  }
}

void LinalgNotationPrinter::visit(const LinalgTransposeNode* op) {
  Precedence precedence = Precedence::TRANSPOSE;
  bool parenthesize =  precedence > parentPrecedence;
  parentPrecedence = precedence;
  if (parenthesize) {
    os << "(";
  }
  op->a.accept(this);
  if (parenthesize) {
    os << ")";
  }
  os << "^T";
}

template <typename Node>
void LinalgNotationPrinter::visitBinary(Node op, Precedence precedence) {
  bool parenthesize =  precedence > parentPrecedence;
  if (parenthesize) {
    os << "(";
  }
  parentPrecedence = precedence;
  op->a.accept(this);
  os << " " << op->getOperatorString() << " ";
  parentPrecedence = precedence;
  op->b.accept(this);
  if (parenthesize) {
    os << ")";
  }
}

void LinalgNotationPrinter::visit(const LinalgAddNode* op) {
  visitBinary(op, Precedence::ADD);
}

void LinalgNotationPrinter::visit(const LinalgSubNode* op) {
  visitBinary(op, Precedence::SUB);
}

void LinalgNotationPrinter::visit(const LinalgMatMulNode* op) {
  visitBinary(op, Precedence::MATMUL);
}

void LinalgNotationPrinter::visit(const LinalgElemMulNode* op) {
  visitBinary(op, Precedence::ELEMMUL);
}

void LinalgNotationPrinter::visit(const LinalgDivNode* op) {
  visitBinary(op, Precedence::DIV);
}

template <class T>
static inline void acceptJoin(LinalgNotationPrinter* printer,
                              std::ostream& stream, const std::vector<T>& nodes,
                              std::string sep) {
  if (nodes.size() > 0) {
    nodes[0].accept(printer);
  }
  for (size_t i = 1; i < nodes.size(); ++i) {
    stream << sep;
    nodes[i].accept(printer);
  }
}

void LinalgNotationPrinter::visit(const LinalgAssignmentNode* op) {
  os << op->lhs.getName() << " " << "= ";
  op->rhs.accept(this);
}

}
