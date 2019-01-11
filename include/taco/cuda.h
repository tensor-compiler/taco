#ifndef CUDA_H
#define CUDA_H

#include <string>
#include <sstream>
#include <ostream>

#ifndef CUDA_BUILT
  #define CUDA_BUILT false
#endif

namespace taco {
/// Functions used by taco to interface with CUDA (especially unified memory)
/// Check if should use CUDA codegen (built and not disabled by user)
bool should_use_CUDA_codegen();
/// Disable CUDA codgen
bool disable_CUDA_codegen();
/// Gets default compiler flags by checking current gpu model
std::string get_default_CUDA_compiler_flags();
/// Allocates memory using unified memory (and checks for errors)
void* cuda_unified_alloc(size_t size);
/// Frees memory from unified memory (and checks for errors)
void cuda_unified_free(void *ptr);

}

#endif
