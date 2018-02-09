#include "taco/expr/expr_printer.h"
#include "taco/expr/expr_nodes.h"

using namespace std;

namespace taco {

ExprPrinter::ExprPrinter(std::ostream& os) : os(os) {
}

void ExprPrinter::print(const IndexExpr& expr) {
  parentPrecedence = Precedence::Top;
  expr.accept(this);
}

void ExprPrinter::visit(const AccessNode* op) {
  os << op->tensorVar.getName() << "(" << util::join(op->indexVars,",") << ")";
}

void ExprPrinter::visit(const NegNode* op) {
  Precedence precedence = Precedence::Neg;
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
  parentPrecedence = Precedence::Func;
  os << "sqrt";
  os << "(";
  op->a.accept(this);
  os << ")";
}

template <typename Node>
void ExprPrinter::visitBinary(Node op, Precedence precedence) {
  bool parenthesize =  precedence > parentPrecedence;
  parentPrecedence = precedence;
  if (parenthesize) {
    os << "(";
  }
  op->a.accept(this);
  os << " " << op->getOperatorString() << " ";
  op->b.accept(this);
  if (parenthesize) {
    os << ")";
  }
}

void ExprPrinter::visit(const AddNode* op) {
  visitBinary(op, Precedence::Add);
}

void ExprPrinter::visit(const SubNode* op) {
  visitBinary(op, Precedence::Sub);
}

void ExprPrinter::visit(const MulNode* op) {
  visitBinary(op, Precedence::Mul);
}

void ExprPrinter::visit(const DivNode* op) {
  visitBinary(op, Precedence::Div);
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
  parentPrecedence = Precedence::Reduction;
  os << ReductionName().get(op->op)
     << "(" << op->var << ")"
     << "(" << op->a << ")";
}


}
