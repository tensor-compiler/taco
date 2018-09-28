#include "taco/codegen/module.h"

#include <iostream>
#include <fstream>
#include <dlfcn.h>
#include <unistd.h>

#include "taco/error.h"
#include "taco/util/strings.h"
#include "taco/util/env.h"
#include "codegen/codegen_c.h"
#include "codegen/codegen_llvm.h"
#include "codegen/llvm_headers.h"

using namespace std;

namespace taco {
namespace ir {

void Module::setJITTmpdir() {
  tmpdir = util::getTmpdir();
}

void Module::setJITLibname() {
  string chars = "abcdefghijkmnpqrstuvwxyz0123456789";
  libname.resize(12);
  for (int i=0; i<12; i++)
    libname[i] = chars[rand() % chars.length()];
}

void Module::addFunction(Stmt func) {
  funcs.push_back(func);
}

void Module::compileToSource(string path, string prefix) {
  std::string sourceSuffix;
  if (!moduleFromUserSource) {
  
    // create a codegen instance and add all the funcs
    bool didGenRuntime = false;
    
    header.str("");
    source.str("");
    header.clear();
    source.clear();
    std::cout << "Generating module " << path << prefix << std::endl;
    if (target.arch == Target::C99) {
      CodeGen_C codegen(source, CodeGen_C::OutputKind::C99Implementation);
      CodeGen_C headergen(header, CodeGen_C::OutputKind::C99Header);
    
    
      for (auto func: funcs) {
        codegen.compile(func, !didGenRuntime);
        headergen.compile(func, !didGenRuntime);
        didGenRuntime = true;
      }
      
      sourceSuffix = ".c";
      
      ofstream source_file;
      source_file.open(path+prefix+sourceSuffix);
      source_file << source.str();
      source_file.close();
      
    } else {
      llvm::LLVMContext context;
      CodeGen_LLVM llvm_codegen(target, context);
      CodeGen_C headergen(header, CodeGen_C::OutputKind::C99Header);
      
      for (auto func: funcs) {
        llvm_codegen.compile(func, !didGenRuntime);
        headergen.compile(func, !didGenRuntime);
        didGenRuntime = true;
      }
      sourceSuffix = ".bc";
      llvm_codegen.writeToFile(path+prefix+sourceSuffix);
    }
    
  }

  
  ofstream header_file;
  header_file.open(path+prefix+".h");
  header_file << header.str();
  header_file.close();
}

void Module::compileToStaticLibrary(string path, string prefix) {
  taco_tassert(false) << "Compiling to a static library is not supported";
}
  
namespace {

void writeShims(vector<Stmt> funcs, string path, string prefix) {
  stringstream shims;
  
  for (auto func: funcs) {
    CodeGen_C::generateShim(func, shims);
  }
  
  ofstream shims_file;
  shims_file.open(path+prefix+"_shims.c");
  shims_file << "#include \"" << path << prefix << ".h\"\n";
  shims_file << shims.str();
  shims_file.close();
}

} // anonymous namespace

string Module::compile() {
  string prefix = tmpdir+libname;
  string fullpath = prefix + ".so";
  
  string cc = util::getFromEnv("TACO_CC", "cc");
  string cflags = util::getFromEnv("TACO_CFLAGS",
    "-O3 -ffast-math -std=c99") + " -shared -fPIC";
  
  string cmd = cc + " " + cflags + " " +
    prefix + (target.arch == Target::X86 ? ".s " : ".c ") +
    (target.arch == Target::C99 ? prefix + "_shims.c " : "") +
    "-o " + prefix + ".so";
  std::cout << "Compiling module " << prefix << std::endl;
  
  // open the output file & write out the source
  compileToSource(tmpdir, libname);
  
  // write out the shims
  if (target.arch == Target::C99) {
    writeShims(funcs, tmpdir, libname);
  }
  
  if (target.arch == Target::X86) {
    // use llc to compile the .ll file
    string llcCommand = util::getFromEnv("TACO_LLC", "llc") +
      " " + prefix + ".bc";
    int err = system(llcCommand.data());
    taco_uassert(err == 0) << "Compilation command failed:\n" << cmd
      << "\nreturned " << err;
  }
  
  // now compile it
  int err = system(cmd.data());
  taco_uassert(err == 0) << "Compilation command failed:\n" << cmd
    << "\nreturned " << err;

  // use dlsym() to open the compiled library
  lib_handle = dlopen(fullpath.data(), RTLD_NOW | RTLD_LOCAL);

  return fullpath;
}

void Module::setSource(string source) {
  this->source << source;
  moduleFromUserSource = true;
}

string Module::getSource() {
  return source.str();
}

void* Module::getFuncPtr(std::string name) {
  return dlsym(lib_handle, name.data());
}

int Module::callFuncPackedRaw(std::string name, void** args) {
  typedef int (*fnptr_t)(void**);
  static_assert(sizeof(void*) == sizeof(fnptr_t),
    "Unable to cast dlsym() returned void pointer to function pointer");
  void* v_func_ptr = getFuncPtr(name);
  fnptr_t func_ptr;
  *reinterpret_cast<void**>(&func_ptr) = v_func_ptr;
  return func_ptr(args);
}

} // namespace ir
} // namespace taco
