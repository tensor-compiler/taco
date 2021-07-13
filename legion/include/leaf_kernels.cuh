#ifndef TACO_LG_CU_LEAF_KERNELS_H
#define TACO_LG_CU_LEAF_KERNELS_H

#include "leaf_kernels.h"
#include "cudalibs.h"
#include "cublas_v2.h"
#include "legion.h"

template<typename T>
__global__
void contractInter(MTTKRPPack pack, T* A, const T* C, const T* inter){
  int C1_dimension = pack.jDim;
  int inter1_dimension = pack.iDim;
  int inter2_dimension = pack.jDim;
  int inter3_dimension = pack.lDim;

  int32_t io = blockIdx.x;
  int32_t ii = (threadIdx.x % (256));
  if (threadIdx.x >= 256) {
    return;
  }

  int32_t f = (io * 256 + ii);
  int32_t i = f / (C1_dimension);
  int32_t iinter = 0 * inter1_dimension + i;
  int32_t iA = i;
  if (i >= inter1_dimension)
    return;

  int32_t j = f % (C1_dimension);
  int32_t jinter = iinter * inter2_dimension + j;
  int32_t jC = j;
  if (j >= C1_dimension)
    return;

  for (int32_t l = 0; l < pack.lDim; l++) {
    int32_t lA = iA * pack.ldA + l;
    int32_t linter = jinter * inter3_dimension + l;
    int32_t lC = jC * pack.ldC + l;
    atomicAdd(&A[lA], inter[linter] * C[lC]);
  }
}

// CUDA version of mttkrp. All buffers must live on memory accessible by the device.
template <typename T>
void cu_mttkrp(MTTKRPPack pack, T* A_vals, const T* B_vals, const T* C_vals, const T* D_vals) {
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

  double alpha = 1.0000000000000000;
  cublasHandle_t handle = getCuBLAS();
  cudaStream_t taskStream = cudaStream_t();
  cudaStreamCreate(&(taskStream));
  CHECK_CUBLAS(cublasSetStream(handle, taskStream));

  // Allocate an intermediate result T(i, j, l).
  T initVal = 0;
  Legion::DeferredBuffer<T, 1> buf(Legion::Memory::Kind::GPU_FB_MEM, Legion::DomainT<1>(Legion::Rect<1>(0, B1_dimension * C1_dimension * D2_dimension - 1)), &initVal);
  T* inter = buf.ptr(0);

  // Perform T(i, j, l) = B(i, j, k) * D(k, l) as a series of GEMM calls.
  for (size_t i = 0; i < B1_dimension; i++) {
    CHECK_CUBLAS(cublasDgemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        D2_dimension,
        C1_dimension,
        D1_dimension,
        &alpha,
        D_vals,
        ldD,
        B_vals + (ldB1 * i),
        ldB3,
        &alpha,
        inter + (C1_dimension * D2_dimension * i),
        D2_dimension
    ));
  }

  // Perform the next reduction A(i, l) = T(i, j, l) * D(j, l).
  contractInter<T><<<(B1_dimension * C1_dimension + 255) / 256, 256, 0, taskStream>>>(pack, A_vals, C_vals, inter);
}

// Small kernel to do a warp level reduction.
template<typename T>
__inline__ __device__
void atomicAddWarp(unsigned mask, T val, T* output) {
  for (int offset = warpSize/2; offset > 0; offset /= 2)  {
    val += __shfl_down_sync(mask, val, offset);
  }
  if (threadIdx.x % warpSize == 0) {
    atomicAdd(output, val);
  }
}

#endif // TACO_LG_CU_LEAF_KERNELS_H
