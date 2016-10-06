#ifndef TACO_BACKEND_C_H
#define TACO_BACKEND_C_H

#include "ir.h"
#include "ir_printer.h"

namespace taco {
namespace internal {

class Module {
public:
  /** Create a module for some source code */
  Module(std::string source);

  /** Compile the source into a library, returning
   * its full path */
  std::string compile();
  
  /** Get a pointer to a compiled function */
  void *get_func(std::string name);

private:
  std::string source;
  std::string libname;
  std::string tmpdir;
  void* lib_handle;
  
  void set_libname();
};

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
} // namespace taco
#endif
