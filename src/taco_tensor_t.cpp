#include "taco/taco_tensor_t.h"

#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>

/*static inline
void init_taco_tensor_t(taco_tensor_t* t, int32_t order, int32_t csize,
                        int32_t* dimensions, int32_t* modeOrdering,
                        taco_mode_t* mode_types) {
  t->order         = order;
  t->dimensions    =     (int32_t*)malloc(order * sizeof(int32_t));
  t->mode_ordering =     (int32_t*)malloc(order * sizeof(int32_t));
  t->mode_types    = (taco_mode_t*)malloc(order * sizeof(taco_mode_t));
  t->indices       =   (uint8_t***)malloc(order * sizeof(uint8_t***));
  t->csize         = csize;

  for (int32_t i = 0; i < order; i++) {
    t->dimensions[i]    = dimensions[i];
    t->mode_ordering[i] = modeOrdering[i];
    t->mode_types[i]    = mode_types[i];
    switch (t->mode_types[i]) {
      case taco_mode_dense:
        t->indices[i]    = (uint8_t**)malloc(1 * sizeof(uint8_t**));
      case taco_mode_sparse:
        t->indices[i]    = (uint8_t**)malloc(2 * sizeof(uint8_t**));
    }
  }
}*/

// https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

taco_tensor_t* init_taco_tensor_t(int32_t order, int32_t csize,
                        int32_t* dimensions, int32_t* modeOrdering,
                        taco_mode_t* mode_types) {
  taco_tensor_t* t;
  gpuErrchk(cudaMallocManaged((void**) &(t), sizeof(taco_tensor_t), 1));
  t->order         = order;
  gpuErrchk(cudaMallocManaged((void**) &(t->dimensions), order * sizeof(int32_t), 1));
  gpuErrchk(cudaMallocManaged((void**) &(t->mode_ordering), order * sizeof(int32_t), 1));
  gpuErrchk(cudaMallocManaged((void**) &(t->mode_types), order * sizeof(taco_mode_t), 1));
  gpuErrchk(cudaMallocManaged((void**) &(t->indices), order * sizeof(uint8_t***), 1));
  t->csize         = csize;

  for (int32_t i = 0; i < order; i++) {
    t->dimensions[i]    = dimensions[i];
    t->mode_ordering[i] = modeOrdering[i];
    t->mode_types[i]    = mode_types[i];
    switch (t->mode_types[i]) {
      case taco_mode_dense:
        gpuErrchk(cudaMallocManaged((void**) &(t->indices[i]), 1 * sizeof(uint8_t**), 1));
      case taco_mode_sparse:
        gpuErrchk(cudaMallocManaged((void**) &(t->indices[i]), 2 * sizeof(uint8_t**), 1));
    }
  }
  return t;
}

/*static inline
void deinit_taco_tensor_t(taco_tensor_t* t) {
  for (int i = 0; i < t->order; i++) {
    free(t->indices[i]);
  }
  free(t->indices);

  free(t->dimensions);
  free(t->mode_ordering);
  free(t->mode_types);
}*/

void deinit_taco_tensor_t(taco_tensor_t* t) {
  for (int i = 0; i < t->order; i++) {
    gpuErrchk(cudaFree(t->indices[i]));
  }
  gpuErrchk(cudaFree(t->indices));

  gpuErrchk(cudaFree(t->dimensions));
  gpuErrchk(cudaFree(t->mode_ordering));
  gpuErrchk(cudaFree(t->mode_types));
  gpuErrchk(cudaFree(t));
}