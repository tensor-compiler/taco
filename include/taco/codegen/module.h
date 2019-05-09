#ifndef TACO_MODULE_H
#define TACO_MODULE_H

#include <map>
#include <vector>
#include <string>
#include <utility>

#include "taco/target.h"
#include "taco/ir/ir.h"

namespace taco {
namespace ir {

class Module {
public:
  /// Create a module for some target
  Module(Target target=getTargetFromEnvironment())
    : lib_handle(nullptr), moduleFromUserSource(false), target(target) {
    setJITLibname();
    setJITTmpdir();
  }

  void reset();

  /// Compile the source into a library, returning its full path
  std::string compile();
  
  /// Compile the module into a source file located at the specified location
  /// path and prefix.  The generated source will be path/prefix.{.c|.bc, .h}
  void compileToSource(std::string path, std::string prefix);
  
  /// Compile the module into a static library located at the specified location
  /// path and prefix.  The generated library will be path/prefix.a
  void compileToStaticLibrary(std::string path, std::string prefix);
  
  /// Add a lowered function to this module */
  void addFunction(Stmt func);

  /// Get the source of the module as a string */
  std::string getSource();
  
  /// Get a function pointer to a compiled function. This returns a void*
  /// pointer, which the caller is required to cast to the correct function type
  /// before calling. If there's no function of this name then a nullptr is
  /// returned.
  void* getFuncPtr(std::string name);

  /// Call a raw function in this module and return the result
  int callFuncPackedRaw(std::string name, void** args);
  
  /// Call a raw function in this module and return the result
  int callFuncPackedRaw(std::string name, std::vector<void*> args) {
    return callFuncPackedRaw(name, args.data());
  }
  
  /// Call a function using the taco_tensor_t interface and return the result
  int callFuncPacked(std::string name, void** args) {
    return callFuncPackedRaw("_shim_"+name, args);
  }
  
  /// Call a function using the taco_tensor_t interface and return the result
  int callFuncPacked(std::string name, std::vector<void*> args) {
    return callFuncPacked(name, args.data());
  }
  
  /// Set the source of the module
  void setSource(std::string source);
  
private:
  std::stringstream source;
  std::stringstream header;
  std::string libname;
  std::string tmpdir;
  void* lib_handle;
  std::vector<Stmt> funcs;
  
  // true iff the module was created from user-provided source
  bool moduleFromUserSource;

  Target target;
  
  void setJITLibname();
  void setJITTmpdir();
};

} // namespace ir
} // namespace taco
#endif
