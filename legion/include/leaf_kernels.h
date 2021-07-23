#ifndef TACO_LG_LEAF_KERNELS_H
#define TACO_LG_LEAF_KERNELS_H

#include <algorithm>
#include <stdint.h>
#include <stddef.h>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include "cblas.h"
#include "legion.h"

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
  size_t ldB1;
  size_t ldB2;
  size_t ldB3;
};

template<typename T>
void mttkrp(MTTKRPPack pack, T* A_vals, const T* B_vals, const T* C_vals, const T* D_vals) {
  size_t B1_dimension = pack.iDim;
  size_t C1_dimension = pack.jDim;
  size_t D1_dimension = pack.kDim;
  size_t D2_dimension = pack.lDim;
  int ldA = pack.ldA;
  int ldC = pack.ldC;
  int ldD = pack.ldD;
  int ldB1 = pack.ldB1;
  int ldB2 = pack.ldB2;
  int ldB3 = pack.ldB3;

  // Allocate an intermediate result T(i, j, l).
  Legion::DeferredBuffer<T, 1> buf(Legion::Memory::Kind::SOCKET_MEM, Legion::DomainT<1>(Legion::Rect<1>(0, B1_dimension * C1_dimension * D2_dimension - 1)));
  T* inter = buf.ptr(0);
  // Initialize the buffer. TODO (rohany): Once CPU fills are fast enough, use the deferred
  // buffer initialization rather than this.
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < B1_dimension; i++) {
    for (size_t j = 0; j < C1_dimension; j++) {
      for (size_t l = 0; l < D2_dimension; l++) {
        inter[(i * C1_dimension + j) * D2_dimension + l] = 0.0;
      }
    }
  }

  // Perform T(i, j, l) = B(i, j, k) * D(k, l) as a series of GEMM calls.
  for (size_t i = 0; i < B1_dimension; i++) {
    cblas_dgemm(
      CblasRowMajor,
      CblasNoTrans,
      CblasNoTrans,
      C1_dimension,
      D2_dimension,
      D1_dimension,
      1.0,
      B_vals + (ldB1 * i),
      ldB3,
      D_vals,
      ldD,
      1.0,
      inter + (C1_dimension * D2_dimension * i),
      D2_dimension
    );
  }

  // Perform the next reduction A(i, l) = T(i, j, l) * C(j, l).
  #pragma omp parallel for schedule(static)
  for (int32_t i = 0; i < B1_dimension; i++) {
    for (int32_t j = 0; j < C1_dimension; j++) {
      int32_t jB = i * C1_dimension + j;
      for (int32_t l = 0; l < D2_dimension; l++) {
        int32_t lA = i * ldA + l;
        int32_t lB = jB * D2_dimension + l;
        int32_t lC = j * ldC + l;
        A_vals[lA] = A_vals[lA] + inter[lB] * C_vals[lC];
      }
    }
  }
}

struct TTVPack {
  int32_t iDim;
  int32_t jDim;
  int32_t kDim;

  size_t ldA;
  size_t ldB2;
  size_t ldB3;
};

template <typename T>
void ttv(TTVPack pack, T* A_vals, const T* B_vals, const T* C_vals) {
  auto iDim = pack.iDim;
  auto jDim = pack.jDim;
  auto kDim = pack.kDim;
  auto ldA = pack.ldA;
  auto ldB2 = pack.ldB2;
  auto ldB3 = pack.ldB3;

  #pragma omp parallel for schedule(static)
  for (int32_t ii = 0; ii < (((iDim) + 3) / 4); ii++) {
    #pragma clang loop interleave(enable) vectorize(enable)
    for (int32_t io = 0; io < 4; io++) {
      int32_t il = ii * 4 + io;
      int32_t i = il;
      if (i >= iDim)
        continue;

      for (int32_t jl = 0; jl < jDim; jl++) {
        int32_t j = jl;
        int32_t jB = i * ldB2 + j;
        int32_t jA = i * ldA + j;

        for (int32_t k = 0; k < kDim; k++) {
          int32_t kB = jB * ldB3 + k;
          A_vals[jA] = A_vals[jA] + B_vals[kB] * C_vals[k];
        }
      }
    }
  }
}

#endif // TACO_LG_LEAF_KERNELS_H
