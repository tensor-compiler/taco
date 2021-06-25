#ifndef TACO_LG_LEAF_KERNELS_H
#define TACO_LG_LEAF_KERNELS_H

#include <stdint.h>
#include <stddef.h>

// An argument pack for MTTKRP.
struct MTTKRPPack {
  // Dimensions for each of the iteration bounds.
  int32_t iDim;
  int32_t jDim;
  int32_t kDim;
  int32_t lDim;
  // Load constants for each tensor.
  size_t ldA;
  size_t ldC;
  size_t ldD;
  size_t ldB2;
  size_t ldB3;
};

template<typename T>
void mttkrp(MTTKRPPack pack, T* A_vals, const T* B_vals, const T* C_vals, const T* D_vals) {
  int B1_dimension = pack.iDim;
  int C1_dimension = pack.jDim;
  int D1_dimension = pack.kDim;
  int D2_dimension = pack.lDim;
  int ldA = pack.ldA;
  int ldC = pack.ldC;
  int ldD = pack.ldD;
  int ldB2 = pack.ldB2;
  int ldB3 = pack.ldB3;

  #pragma omp parallel for schedule(static)
  for (int32_t io = 0; io < ((B1_dimension + 3) / 4); io++) {
    #pragma clang loop interleave(enable) vectorize(enable)
    for (int32_t ii = 0; ii < 4; ii++) {
      int32_t i = io * 4 + ii;
      if (i >= B1_dimension)
        continue;

      for (int32_t j = 0; j < C1_dimension; j++) {
        int32_t jB = i * ldB2 + j;
        for (int32_t k = 0; k < D1_dimension; k++) {
          int32_t kB = jB * ldB3 + k;
          for (int32_t l = 0; l < D2_dimension; l++) {
            int32_t lA = i * ldA + l;
            int32_t lC = j * ldC + l;
            int32_t lD = k * ldD + l;
            A_vals[lA] = A_vals[lA] + (B_vals[kB] * C_vals[lC]) * D_vals[lD];
          }
        }
      }
    }
  }
}

#endif // TACO_LG_LEAF_KERNELS_H