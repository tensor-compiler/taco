#include "taco/cuda.h"
#include "taco/error.h"

using namespace std;
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

string get_default_CUDA_compiler_flags() {
#if CUDA_BUILT
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  string computeCap = to_string(prop.major) + to_string(prop.minor);
  return "-w -O3 -Xcompiler \"-fPIC -shared -ffast-math -O3\" --generate-code arch=compute_" + computeCap + ",code=sm_" + computeCap;
#else
  taco_ierror;
  return "";
#endif
}

#if CUDA_BUILT
  // https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
  #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
  inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
  {
    if (code != cudaSuccess && code != 29) // 29 is driver is shutting down which is normal behavior 
    {
      taco_ierror << "GPUassert: " << code << " " << cudaGetErrorString(code) << " " << file << " " << line;
    }
  }
#endif

void* cuda_unified_alloc(size_t size) {
  #if CUDA_BUILT
    void *ptr;
    gpuErrchk(cudaMallocManaged(&ptr, size, 1));
    return ptr;
  #else
    taco_ierror;
    return nullptr;
  #endif
}

void cuda_unified_free(void *ptr) {
  #if CUDA_BUILT
    gpuErrchk(cudaFree(ptr));
  #else
    taco_ierror;
  #endif
}
}