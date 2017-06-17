#ifndef TACO_IR_PRINTER_H
#define TACO_IR_PRINTER_H

#include <ostream>
#include <map>
#include <string>
#include "taco/ir/ir_visitor.h"
#include "taco/util/scopedmap.h"
#include "taco/util/name_generator.h"

namespace taco {
namespace ir {

class IRPrinter : public IRVisitorStrict {
public:
  IRPrinter(std::ostream& stream);
  IRPrinter(std::ostream& stream, bool color, bool simplify);
  virtual ~IRPrinter();

  void print(Stmt);

protected:
  virtual void visit(const Literal*);
  virtual void visit(const Var*);
  virtual void visit(const Neg*);
  virtual void visit(const Sqrt*);
  virtual void visit(const Add*);
  virtual void visit(const Sub*);
  virtual void visit(const Mul*);
  virtual void visit(const Div*);
  virtual void visit(const Rem*);
  virtual void visit(const Min*);
  virtual void visit(const Max*);
  virtual void visit(const BitAnd*);
  virtual void visit(const Eq*);
  virtual void visit(const Neq*);
  virtual void visit(const Gt*);
  virtual void visit(const Lt*);
  virtual void visit(const Gte*);
  virtual void visit(const Lte*);
  virtual void visit(const And*);
  virtual void visit(const Or*);
  virtual void visit(const IfThenElse*);
  virtual void visit(const Case*);
  virtual void visit(const Load*);
  virtual void visit(const Store*);
  virtual void visit(const For*);
  virtual void visit(const While*);
  virtual void visit(const Block*);
  virtual void visit(const Scope*);
  virtual void visit(const Function*);
  virtual void visit(const VarAssign*);
  virtual void visit(const Allocate*);
  virtual void visit(const Comment*);
  virtual void visit(const BlankLine*);
  virtual void visit(const Print*);
  virtual void visit(const GetProperty*);

  std::ostream &stream;
  int indent;
  bool color;
  bool simplify;
  bool omitNextParen;

  util::NameGenerator varNameGenerator;
  util::ScopedMap<Expr, std::string> varNames;

  void resetNameCounters();

  void doIndent();
  void printBinOp(Expr a, Expr b, std::string op);

  std::string keywordString(std::string);
  std::string commentString(std::string);
};

}
}

#endif
