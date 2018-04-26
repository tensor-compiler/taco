#include "taco/index_notation/expr_printer.h"
#include "taco/index_notation/expr_nodes.h"

using namespace std;

namespace taco {

ExprPrinter::ExprPrinter(std::ostream& os) : os(os) {
}

void ExprPrinter::print(const IndexExpr& expr) {
  parentPrecedence = Precedence::TOP;
  expr.accept(this);
}

void ExprPrinter::print(const TensorExpr& expr) {
  parentPrecedence = Precedence::TOP;
  expr.accept(this);
}

void ExprPrinter::visit(const AccessNode* op) {
  os << op->tensorVar.getName() << "(" << util::join(op->indexVars,",") << ")";
}

void ExprPrinter::visit(const NegNode* op) {
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

void ExprPrinter::visit(const SqrtNode* op) {
  parentPrecedence = Precedence::FUNC;
  os << "sqrt";
  os << "(";
  op->a.accept(this);
  os << ")";
}

template <typename Node>
void ExprPrinter::visitBinary(Node op, Precedence precedence) {
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

void ExprPrinter::visit(const AddNode* op) {
  visitBinary(op, Precedence::ADD);
}

void ExprPrinter::visit(const SubNode* op) {
  visitBinary(op, Precedence::SUB);
}

void ExprPrinter::visit(const MulNode* op) {
  visitBinary(op, Precedence::MUL);
}

void ExprPrinter::visit(const DivNode* op) {
  visitBinary(op, Precedence::DIV);
}

template <typename Node>
void ExprPrinter::visitImmediate(Node op) {
  os << op->val;
}

void ExprPrinter::visit(const IntImmNode* op) {
  visitImmediate(op);
}

void ExprPrinter::visit(const FloatImmNode* op) {
  visitImmediate(op);
}

void ExprPrinter::visit(const ComplexImmNode* op) {
  visitImmediate(op);
}

void ExprPrinter::visit(const UIntImmNode* op) {
  visitImmediate(op);
}

void ExprPrinter::visit(const ReductionNode* op) {
  struct ReductionName : ExprVisitor {
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

void ExprPrinter::visit(const AssignmentNode* op) {
  op->lhs.accept(this);
  os << " = ";
  op->rhs.accept(this);
}

void ExprPrinter::visit(const ForallNode* op) {
  os << "forall(" << op->indexVar << ", ";
  op->expr.accept(this);
  os << ")";
}

void ExprPrinter::visit(const WhereNode* op) {
  os << "where(";
  op->consumer.accept(this);
  os << ", ";
  op->producer.accept(this);
  os << ")";
}

}
