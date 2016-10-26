#ifndef TACO_BACKEND_C_H
#define TACO_BACKEND_C_H

#include <map>
#include <vector>

#include "ir.h"
#include "ir_printer.h"

namespace taco {
namespace ir {

class Module {
public:
  /** Create a module for some source code */
  Module(std::string source);

  /** Compile the source into a library, returning
   * its full path */
  std::string compile();
  
  /** Get a function pointer to a compiled function.
   * This returns a void* pointer, which the caller is
   * required to cast to the correct function type before
   * calling.
   */
  void *get_func(std::string name);
  
  /** Call a function in this module and return the result */
  template <typename... Args>
  int call_func(std::string name, Args... args) {
    typedef int (*fnptr_t)(Args...);
    fnptr_t func_ptr = (fnptr_t)get_func(name);
    return func_ptr(args...);
  }
  
  /** Call a function in this module and return the result */
  int call_func_packed(std::string name, void** args);
  
  int call_func_packed(std::string name, std::vector<void*> args) {
    return call_func_packed(name, &(args[0]));
  }
  
private:
  std::string source;
  std::string libname;
  std::string tmpdir;
  void* lib_handle;
  
  void set_libname();
};

class CodeGen_C : public IRPrinterBase {
public:
  /** Initialize a code generator that generates code to an
   * output stream.
   */
  CodeGen_C(std::ostream &dest);
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
  
  bool func_block;
  std::string func_decls;
  
  static int unique_name_counter;
  
  std::map<Expr, std::string, ExprCompare> var_map;
  std::ostream &out;

};

} // namespace ir
} // namespace taco
#endif
