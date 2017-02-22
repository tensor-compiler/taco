#include <iostream>
#include <fstream>
#include <dlfcn.h>
#include <unistd.h>


#include "module.h"
#include "util/strings.h"
#include "error.h"

using namespace std;

namespace taco {
namespace ir {

namespace {

string get_from_env(string flag, string dflt) {
  char const *ret = getenv(flag.c_str());
  if (!ret) {
    return dflt;
  } else {
    return string(ret);
  }
}

}

void Module::set_jit_tmpdir() {
  // use POSIX logic for finding a temp dir
  auto tmp = get_from_env("TMPDIR", "/tmp/");

  uassert(access(tmp.c_str(), W_OK) == 0) <<
    "Unable to write to temporary directory for code generation. "
    "Please set the environment variable TMPDIR to somewhere writable";
  
  tmpdir = tmp;
}

void Module::set_jit_libname() {
  string chars = "abcdefghijkmnpqrstuvwxyz0123456789";
  libname.resize(12);
  for (int i=0; i<12; i++)
    libname[i] = chars[rand() % chars.length()];
}

void Module::add_function(Stmt func) {
  codegen->compile(func);
}

string Module::compile() {
  string prefix = tmpdir+libname;
  string fullpath = prefix + ".so";
  
  string cc = get_from_env("TACO_CC", "cc");
  string cflags = get_from_env("TACO_CFLAGS",
    "-O3 -ffast-math -std=c99 -shared -fPIC");
  
  string cmd = cc + " " + cflags + " " +
    prefix + ".c " +
    "-o " + prefix + ".so";

  // open the output file & write out the source
  ofstream source_file;
  source_file.open(prefix+".c");
  source_file << source.str();
  source_file.close();
  
  // now compile it
  int err = system(cmd.data());
  uassert(err == 0) << "Compilation command failed:\n" << cmd
    << "\nreturned " << err;

  // use dlsym() to open the compiled library
  lib_handle = dlopen(fullpath.data(), RTLD_NOW | RTLD_LOCAL);

  return fullpath;
}

string Module::get_source() {
  return source.str();
}

void* Module::get_func(std::string name) {
  void* ret = dlsym(lib_handle, name.data());
  uassert(ret != nullptr) << "Function " << name << " not found in module " <<
    tmpdir << libname;
  return ret;
}

int Module::call_func_packed(std::string name, void** args) {
  typedef int (*fnptr_t)(void**);
  static_assert(sizeof(void*) == sizeof(fnptr_t),
    "Unable to cast dlsym() returned void pointer to function pointer");
  void* v_func_ptr = get_func(name);
  fnptr_t func_ptr;
  *reinterpret_cast<void**>(&func_ptr) = v_func_ptr;
  return func_ptr(args);
}

} // namespace ir
} // namespace taco
