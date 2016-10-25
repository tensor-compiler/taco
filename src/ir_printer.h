#ifndef TACO_IR_PRINTER_H
#define TACO_IR_PRINTER_H

#include <ostream>
#include "ir_visitor.h"

namespace taco {
namespace ir {

class IRPrinterBase : public IRVisitor {
public:
  /** Construct an IRPrinter using a specific output stream */
  IRPrinterBase(std::ostream &);
  virtual ~IRPrinterBase();
  
protected:
  std::ostream &stream;
  int indent;
  void do_indent();
  
  void print_binop(Expr a, Expr b, std::string op);
  
  virtual void visit(const Literal*);
  virtual void visit(const Var*);
  virtual void visit(const Add*);
  virtual void visit(const Sub*);
  virtual void visit(const Mul*);
  virtual void visit(const Div*);
  virtual void visit(const Rem*);
  virtual void visit(const Min*);
  virtual void visit(const Max*);
  virtual void visit(const Eq*);
  virtual void visit(const Neq*);
  virtual void visit(const Gt*);
  virtual void visit(const Lt*);
  virtual void visit(const Gte*);
  virtual void visit(const Lte*);
  virtual void visit(const And*);
  virtual void visit(const Or*);
  virtual void visit(const IfThenElse*);
  virtual void visit(const Load*);
  virtual void visit(const Store*);
  virtual void visit(const For*);
  virtual void visit(const While*);
  virtual void visit(const Block*);
  virtual void visit(const Function*);
  virtual void visit(const VarAssign*);
  virtual void visit(const Allocate*);
  virtual void visit(const Comment*);
  virtual void visit(const BlankLine*);
  virtual void visit(const Print*);
  virtual void visit(const GetProperty*);
};

class IRPrinter : public IRPrinterBase {
public:
  IRPrinter(std::ostream &stream) : IRPrinterBase(stream) { }
  virtual ~IRPrinter();

  using IRPrinterBase::visit;
  virtual void visit(const Function*);
  virtual void visit(const For*);
  virtual void visit(const Block*);
};

}
}

#endif
