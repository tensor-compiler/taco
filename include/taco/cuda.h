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
/// Check if should use CUDA codegen
bool should_use_CUDA_codegen();
/// Check if should use CUDA unified memory
bool should_use_CUDA_unified_memory();
/// Enable/Disable CUDA codegen
void set_CUDA_codegen_enabled(bool enabled);
/// Enable/Disable CUDA unified memory
void set_CUDA_unified_memory_enabled(bool enabled);
/// Gets default compiler flags by checking current gpu model
std::string get_default_CUDA_compiler_flags();
/// Allocates memory using unified memory (and checks for errors)
void* cuda_unified_alloc(size_t size);
/// Frees memory from unified memory (and checks for errors)
void cuda_unified_free(void *ptr);

}

#endif
