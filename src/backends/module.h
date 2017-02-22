#ifndef TACO_MODULE_H
#define TACO_MODULE_H

#include <map>
#include <vector>
#include <string>

#include "target.h"
#include "ir.h"
#include "codegen_c.h"

namespace taco {
namespace ir {

class Module {
public:
  /** Create a module for some target */
  Module(Target target=get_target_from_environment()) : target(target) {
    set_jit_libname();
    set_jit_tmpdir();
    
    iassert(target.arch == Target::C99) << "Only C99 codegen supported currently";
    codegen = new CodeGen_C(source);
  }
  
  
  ~Module() { delete codegen; }

  /** Compile the source into a library, returning
   * its full path 
   */
  std::string compile();
  
  /** Compile the module into a source file located
   * at the specified location path and prefix.  The generated
   * source will be path/prefix.{.c|.bc, .h}
   */
  void compile_to_source(std::string path, std::string prefix);
  
  /** Compile the module into a static library located
   * at the specified location path and prefix.  The generated
   * library will be path/prefix.a
   */
  void compile_to_static_library(std::string path, std::string prefix);
  
  /** Add a lowered function to this module */
  void add_function(Stmt func);
  
  /** Get the source of the module as a string */
  std::string get_source();
  
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
    static_assert(sizeof(void*) == sizeof(fnptr_t),
      "Unable to cast dlsym() returned void pointer to function pointer");
    void* v_func_ptr = get_func(name);
    fnptr_t func_ptr;
    *reinterpret_cast<void**>(&func_ptr) = v_func_ptr;
    return func_ptr(args...);
  }
  
  /** Call a function in this module and return the result */
  int call_func_packed(std::string name, void** args);
  
  int call_func_packed(std::string name, std::vector<void*> args) {
    return call_func_packed(name, &(args[0]));
  }
  
private:
  std::stringstream source;
  std::string libname;
  std::string tmpdir;
  void* lib_handle;

  Target target;
  CodeGen_C *codegen;  // TODO: replace with superclass
  
  void set_jit_libname();
  void set_jit_tmpdir();
};

} // namespace ir
} // namespace taco
#endif
