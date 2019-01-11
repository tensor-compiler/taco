#include "taco/cuda.h"

namespace taco {
/// Functions used by taco to interface with CUDA (especially unified memory)
static bool enable_CUDA_codegen = true;
bool should_use_CUDA_codegen() {
  return CUDA_BUILT && enable_CUDA_codegen;
}

bool disable_CUDA_codegen() {
  enable_CUDA_codegen = false;
  return CUDA_BUILT;
}
}