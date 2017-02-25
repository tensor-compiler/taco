#ifndef TACO_BACKEND_C_H
#define TACO_BACKEND_C_H

#include <map>
#include <vector>

#include "ir/ir.h"
#include "ir/ir_printer.h"

namespace taco {
namespace ir {


class CodeGen_C : public IRPrinterBase {
public:
  /** Kind of output: header or implementation */
  enum OutputKind { C99Header, C99Implementation };

  /** Initialize a code generator that generates code to an
   * output stream.
   */
  CodeGen_C(std::ostream &dest, OutputKind output_kind);
  ~CodeGen_C();
  
  /** Compile a lowered function */
  void compile(Stmt stmt);

  static std::string gen_unique_name(std::string var_name="");
  
protected:
  using IRPrinterBase::visit;
  void visit(const Function*);
  void visit(const Var*);
  void visit(const For*);
  void visit(const While*);
  void visit(const Block*);
  void visit(const GetProperty*);
  void visit(const Case*);
  void visit(const Min*);
  void visit(const Allocate*);
  void visit(const Sqrt*);
  
  bool func_block;
  std::string func_decls;
  
  static int unique_name_counter;
  
  std::map<Expr, std::string, ExprCompare> var_map;
  std::ostream &out;
  
  OutputKind output_kind;

};

} // namespace ir
} // namespace taco
#endif
