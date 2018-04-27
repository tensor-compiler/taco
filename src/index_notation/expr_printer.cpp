#include "taco/index_notation/expr_printer.h"
#include "taco/index_notation/expr_nodes.h"

using namespace std;

namespace taco {

IndexNotationPrinter::IndexNotationPrinter(std::ostream& os) : os(os) {
}

void IndexNotationPrinter::print(const IndexExpr& expr) {
  parentPrecedence = Precedence::TOP;
  expr.accept(this);
}

void IndexNotationPrinter::print(const IndexStmt& expr) {
  parentPrecedence = Precedence::TOP;
  expr.accept(this);
}

void IndexNotationPrinter::visit(const AccessNode* op) {
  os << op->tensorVar.getName() << "(" << util::join(op->indexVars,",") << ")";
}

void IndexNotationPrinter::visit(const NegNode* op) {
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

void IndexNotationPrinter::visit(const SqrtNode* op) {
  parentPrecedence = Precedence::FUNC;
  os << "sqrt";
  os << "(";
  op->a.accept(this);
  os << ")";
}

template <typename Node>
void IndexNotationPrinter::visitBinary(Node op, Precedence precedence) {
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

void IndexNotationPrinter::visit(const AddNode* op) {
  visitBinary(op, Precedence::ADD);
}

void IndexNotationPrinter::visit(const SubNode* op) {
  visitBinary(op, Precedence::SUB);
}

void IndexNotationPrinter::visit(const MulNode* op) {
  visitBinary(op, Precedence::MUL);
}

void IndexNotationPrinter::visit(const DivNode* op) {
  visitBinary(op, Precedence::DIV);
}

template <typename Node>
void IndexNotationPrinter::visitImmediate(Node op) {
  os << op->val;
}

void IndexNotationPrinter::visit(const IntImmNode* op) {
  visitImmediate(op);
}

void IndexNotationPrinter::visit(const FloatImmNode* op) {
  visitImmediate(op);
}

void IndexNotationPrinter::visit(const ComplexImmNode* op) {
  visitImmediate(op);
}

void IndexNotationPrinter::visit(const UIntImmNode* op) {
  visitImmediate(op);
}

void IndexNotationPrinter::visit(const ReductionNode* op) {
  struct ReductionName : IndexNotationVisitor {
    std::string reductionName;
    std::string get(IndexExpr expr) {
      expr.accept(this);
      return reductionName;
    }
    void visit(const AddNode* node) {
      reductionName = "sum";
    }
    void visit(const MulNode* node) {
      reductionName = "product";
    }
    void visit(const BinaryExprNode* node) {
      reductionName = "reduction(" + node->getOperatorString() + ")";
    }
  };
  parentPrecedence = Precedence::REDUCTION;
  os << ReductionName().get(op->op) << "(" << op->var << ", ";
  op->a.accept(this);
  os << ")";
}

void IndexNotationPrinter::visit(const AssignmentNode* op) {
  struct OperatorName : IndexNotationVisitor {
    std::string operatorName;
    std::string get(IndexExpr expr) {
      if (!expr.defined()) return "";
      expr.accept(this);
      return operatorName;
    }
    void visit(const BinaryExprNode* node) {
      operatorName = node->getOperatorString();
    }
  };

  op->lhs.accept(this);
  os << " " << OperatorName().get(op->op) << "= ";
  op->rhs.accept(this);
}

void IndexNotationPrinter::visit(const ForallNode* op) {
  os << "forall(" << op->indexVar << ", ";
  op->stmt.accept(this);
  os << ")";
}

void IndexNotationPrinter::visit(const WhereNode* op) {
  os << "where(";
  op->consumer.accept(this);
  os << ", ";
  op->producer.accept(this);
  os << ")";
}

}
