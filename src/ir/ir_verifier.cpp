#include "taco/ir/ir_verifier.h"
#include "taco/ir/ir_visitor.h"
#include "taco/error/error_messages.h"

namespace taco {
namespace ir {

namespace {
class IRVerifier : IRVisitor {
public:
  std::stringstream messages;
  void verify(const Expr e) {
    e.accept(this);
  }
  void verify(const Stmt s) {
    s.accept(this);
  }
protected:
  using IRVisitor::visit;

  template <typename C>
  void verify_operand_types_consistent(C op) {
    if (op->a.type() != op->b.type()) {
      messages << "Node: " << (Expr)op << " has operands with different types (" 
               << op->a.type() << " vs. " << op->b.type() << ")\n";
    }
  }
  
  template <typename C>
  void verify_operand_types_consistent_with_result(C op) {
    auto tp = op->type;
    if (op->a.type() != tp) {
      messages << "Node: " << (Expr)op << " has left operand with different "
               << "type from result (expected " << tp << " but got " 
               << op->a.type() << ")\n";
    }
    if (op->b.type() != tp) {
      messages << "Node: " << (Expr)op << " has right operand with different "
               << "type from result (expected " << tp << " but got " 
               << op->b.type() << ")\n";
    }
  }

  void visit(const Add *op) {
    verify_operand_types_consistent_with_result(op);
    op->a.accept(this);
    op->b.accept(this);
  }

  void visit(const Sub *op) {
    verify_operand_types_consistent_with_result(op);
    op->a.accept(this);
    op->b.accept(this);
  }

  void visit(const Mul *op) {
    verify_operand_types_consistent_with_result(op);
    op->a.accept(this);
    op->b.accept(this);
  }

  void visit(const Div *op) {
    verify_operand_types_consistent_with_result(op);
    op->a.accept(this);
    op->b.accept(this);
  }

  void visit(const Rem *op) {
    messages << "Node: " << (Expr)op << " is deprecated\n";
  }
  
  void visit(const Min *op) {
    auto tp = op->type;
    for (auto &x: op->operands) {
      if (x.type() != tp) {
        messages << "Node: " << (Expr)op << " has operand with incorrect type\n";
      }
      x.accept(this);
    }
  }
  
  void visit(const Max *op) {
    verify_operand_types_consistent_with_result(op);
    op->a.accept(this);
    op->b.accept(this);
  }
  
  void visit(const BitAnd *op) {
    // TODO: do we want to enforce integer-ness?
    verify_operand_types_consistent_with_result(op);
    op->a.accept(this);
    op->b.accept(this);
  }

  void visit(const BitOr *op) {
    // TODO: do we want to enforce integer-ness?
    verify_operand_types_consistent_with_result(op);
    op->a.accept(this);
    op->b.accept(this);
  }

  void visit(const Eq *op) {
    verify_operand_types_consistent(op);
    op->a.accept(this);
    op->b.accept(this);
  }

  void visit(const Neq *op) {
    verify_operand_types_consistent(op);
    op->a.accept(this);
    op->b.accept(this);
  }

  void visit(const Gt *op) {
    verify_operand_types_consistent(op);
    op->a.accept(this);
    op->b.accept(this);
  }

  void visit(const Lt *op) {
    verify_operand_types_consistent(op);
    op->a.accept(this);
    op->b.accept(this);
  }
  
  void visit(const Gte *op) {
    verify_operand_types_consistent(op);
    op->a.accept(this);
    op->b.accept(this);
  }
  
  void visit(const Lte* op) {
    verify_operand_types_consistent(op);
    op->a.accept(this);
    op->b.accept(this);
  }

  void visit(const And *op) {
    auto tp = op->type;
    if (op->a.type() != tp || op->b.type() != tp || !tp.isBool()) {
      messages << "Node: " << (Expr)op << " has operand with incorrect type\n";
    }
    op->a.accept(this);
    op->b.accept(this);
  }
  
  void visit(const Or *op) {
    auto tp = op->type;
    if (op->a.type() != tp || op->b.type() != tp ||
        !tp.isBool()) {
      messages << "Node: " << (Expr)op << " has operand with incorrect type\n";
    }
    op->a.accept(this);
    op->b.accept(this);
  }

  void visit(const IfThenElse *op) {
    if (!op->cond.type().isBool()) {
      messages << "Node: " << (Stmt)op << " has condition "
        << (Expr)op->cond << " that is not boolean\n";
    }
    op->cond.accept(this);
    op->then.accept(this);

    if (op->otherwise.defined()) {
      op->otherwise.accept(this);
    }
  }

  void visit(const Case *op) {
    for (auto &c : op->clauses) {
      if (!c.first.type().isBool()) {
        messages << "Node: " << (Stmt)op << " has condition "
          << (Expr)c.first << " that is not boolean\n";
      }
      c.first.accept(this);
      c.second.accept(this);
    }
  }

  void visit(const Switch *op) {
    // the control statement must be of integer type
    if (!op->controlExpr.type().isInt() && !op->controlExpr.type().isUInt()) {
      messages << "Node: " << (Stmt)op
        << " has a control statement with non-integer type\n";
    }
    op->controlExpr.accept(this);
    
    for (auto &x: op->cases) {
      if (!(x.first.as<Literal>()) ||
          (!x.first.type().isInt() && !x.first.type().isUInt())) {
        messages << "Node: " << (Stmt)op
          << " has clauses with non-integer literal switch values\n";
      }
      x.second.accept(this);
    }
  }
  
  void visit(const Load *op) {
    if (op->type != op->arr.type()) {
      messages << "Node: " << (Expr)op
        << " has type that differs from the target array (" << op->type
        << " vs. " << op->arr.type() << ")\n";
    }
    op->arr.accept(this);
    op->loc.accept(this);
  }
  
  void visit(const Store *op) {
    auto tp = op->arr.type();
    if (tp != op->data.type()) {
      messages << "Node: " << (Stmt)op
        << " is storing data of different type from array (" << op->data.type() 
        << " vs. " << tp << ")\n";
    }
    op->arr.accept(this);
    op->data.accept(this);
    op->loc.accept(this);
  }
  
  void visit(const For *op) {
    auto loopVarType = op->var.type();
    if (op->start.type() != loopVarType) {
      messages << "Node: " << (Stmt)op << " has start with different type from "
               << "loop variable (expected " << loopVarType << " but got " 
               << op->start.type() << ")\n";
    }
    if (op->end.type() != loopVarType) {
      messages << "Node: " << (Stmt)op << " has end with different type from "
               << "loop variable (expected " << loopVarType << " but got " 
               << op->end.type() << ")\n";
    }
    if (op->increment.type() != loopVarType) {
      messages << "Node: " << (Stmt)op << " has increment with different type "
               << "from loop variable (expected " << loopVarType << " but got " 
               << op->increment.type() << ")\n";
    }
    if (!op->var.as<Var>()) {
      messages << "Node: " << (Stmt)op << " loop variable is not Var\n";
    }
    op->start.accept(this);
    op->end.accept(this);
    op->increment.accept(this);
    op->contents.accept(this);
  }
  
  void visit(const While *op) {
    if (!op->cond.type().isBool()) {
      messages << "Node: " << (Stmt)op << " has non-boolean condition\n";
    }
    op->cond.accept(this);
    op->contents.accept(this);
  }

  void visit(const VarDecl *op) {
    if (!op->var.as<Var>()) {
      messages << "Node: " << (Stmt)op << " must have Var node on lhs\n";
    }
    if (op->var.type() != op->rhs.type()) {
      messages << "Node: " << (Stmt)op
               << " has different types on rhs and lhs\n";
    }
    op->var.accept(this);
    op->rhs.accept(this);
  }

  void visit(const Assign *op) {
    if (!op->lhs.as<Var>() && !op->lhs.as<GetProperty>()) {
      messages << "Node: " << (Stmt)op
               << " must have Var or GetProperty node on lhs\n";
    }
    if (op->lhs.type() != op->rhs.type()) {
      messages << "Node: " << (Stmt)op << " has different types on rhs and lhs\n";
    }
    op->lhs.accept(this);
    op->rhs.accept(this);
  }
  
  void visit(const Allocate *op) {
    if (!op->var.as<Var>() && !op->var.as<GetProperty>()) {
      messages << "Node: " << (Stmt)op << " must have Var node on lhs\n";
    }
    op->num_elements.accept(this);
  }

  void visit(const Print *op) {
    // probably should check that the format string is correct for the
    // parameters in the node.
    for (auto &p: op->params) {
      p.accept(this);
    }
  }
  
  void visit(const GetProperty *op) {
    op->tensor.accept(this);
    // nothing here yet, but might be required
  }

};

} // anonymous namespace

bool verify(const Expr e, std::string *messages) {
  using namespace std;
  INIT_REASON(messages);
  
  IRVerifier verifier;
  verifier.verify(e);
  *messages = verifier.messages.str();
  
  return !(*messages).empty();
}

bool verify(const Stmt s, std::string *messages) {
  using namespace std;
  INIT_REASON(messages);
  
  IRVerifier verifier;
  verifier.verify(s);
  *messages = verifier.messages.str();
  
  return !(*messages).empty();
}

} // namespace ir
} // namespace taco
