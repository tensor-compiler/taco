#ifndef LLVM_H
#define LLVM_H

#ifndef USE_LLVM
  #define USE_LLVM false
#endif

namespace taco {
bool should_use_LLVM_codegen();
void set_LLVM_codegen_enabled(bool enabled);
}

#endif