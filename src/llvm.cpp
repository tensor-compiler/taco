#include "taco/llvm.h"

namespace taco{

static bool LLVM_codegen_enabled = USE_LLVM;
bool should_use_LLVM_codegen(){
  return LLVM_codegen_enabled;
}

void set_LLVM_codegen_enabled(bool enabled) {
  LLVM_codegen_enabled = enabled;
}

} // namespace taco