#ifndef TACIT_IR_PRINTER_H
#define TACIT_IR_PRINTER_H

#include <ostream>
#include "ir_visitor.h"

namespace tacit {
namespace internal {

class IRPrinter : public IRVisitor {
public:
  /** Construct an IRPrinter using a specific output stream */
  IRPrinter(std::ostream &);
  
protected:
  std::ostream &stream;
  int indent;
  void do_indent();
  
  void print_binop(Expr a, Expr b, std::string op);
  
  void visit(const Literal*);
  void visit(const Var*);
  void visit(const Add*);
  void visit(const Sub*);
  void visit(const Mul*);
  void visit(const Div*);
  void visit(const Rem*);
  void visit(const Min*);
  void visit(const Max*);
  void visit(const Eq*);
  void visit(const Neq*);
  void visit(const Gt*);
  void visit(const Lt*);
  void visit(const Gte*);
  void visit(const Lte*);
  void visit(const And*);
  void visit(const Or*);
  void visit(const IfThenElse*);
  void visit(const Load*);
  void visit(const Store*);
  void visit(const For*);
  void visit(const Block*);

};


}
}

#endif
