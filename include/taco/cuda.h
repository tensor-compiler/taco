#ifndef CUDA_H
#define CUDA_H

#include <string>
#include <sstream>
#include <ostream>

#ifndef CUDA_BUILT
#define CUDA_BUILT true // TODO
#endif

namespace taco {
/// Functions used by taco to interface with CUDA (especially unified memory)
bool should_use_CUDA_codegen();

bool disable_CUDA_codegen();
}

#endif
