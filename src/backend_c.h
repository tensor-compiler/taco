#ifndef TACIT_BACKEND_C_H
#define TACIT_BACKEND_C_H

#include "ir.h"
#include "ir_printer.h"

namespace tacit {
namespace internal {

class CodeGen_C : public IRPrinter {
public:
  /** Initialize a code generator that generates code to an
   * output stream.
   */
  CodeGen_C(std::ostream &dest);
  ~CodeGen_C();
  
  /** Compile a lowered function */
  void compile(const Function* func);

  static std::string gen_unique_name(std::string var_name="");
  
protected:
  void visit(const Var*);
  void visit(const For*);
  void visit(const Block*);
  
  bool func_block;
  std::string func_decls;
  
  static int unique_name_counter;
  
  std::map<Expr, std::string, ExprCompare> var_map;
  std::ostream &out;
};

} // namespace internal
} // namespace tacit
#endif
