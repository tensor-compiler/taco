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

Module::Module(string source) : source(source) {
  
  // use POSIX logic for finding a temp dir
  char const *tmp = getenv("TMPDIR");
  if (!tmp) {
    tmp = "/tmp/";
  }
  
  uassert(access(tmp, W_OK) == 0) <<
    "Unable to write to temporary directory for code generation. "
    "Please set the environment variable TMPDIR to somewhere writable";
  
  tmpdir = tmp;
  
  // set the library name to some random alphanum string
  set_libname();
}

void Module::set_libname() {
  string chars = "abcdefghijkmnpqrstuvwxyz0123456789";
  libname.resize(12);
  for (int i=0; i<12; i++)
    libname[i] = chars[rand() % chars.length()];
}

string Module::compile() {
  string prefix = tmpdir+libname;
  string fullpath = prefix + ".so";
  
  string cmd = "cc -O3 -ffast-math -std=c99 -shared -fPIC " +
    prefix + ".c " +
    "-o " + prefix + ".so";

  // open the output file & write out the source
  ofstream source_file;
  source_file.open(prefix+".c");
  source_file << source;
  source_file.close();
  
  // now compile it
//  cout << "Executing " << cmd << endl;
  int err = system(cmd.data());
  uassert(err == 0) << "Compilation command failed:\n" << cmd
    << "\nreturned " << err;

  // use dlsym() to open the compiled library
  lib_handle = dlopen(fullpath.data(), RTLD_NOW | RTLD_LOCAL);

  return fullpath;
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
