#ifndef TACO_LG_CU_LEAF_KERNELS_H
#define TACO_LG_CU_LEAF_KERNELS_H

#include "leaf_kernels.h"
#include "cudalibs.h"
#include "cublas_v2.h"
#include "legion.h"

// This atomicAddWarp kernel ensures that the warp level reduction only
// happens if all threads in the warp are indeed writing to the same
// output location.
template<typename T>
__device__ inline void atomicAddWarp(T *output, int index, T val)
{
  int leader_index = __shfl_sync(__activemask(), index, 0);
  int mask = __ballot_sync(__activemask(), leader_index == index);
  if(mask == __activemask()) {
    val += __shfl_down_sync(__activemask(), val, 16);
    val += __shfl_down_sync(__activemask(), val, 8);
    val += __shfl_down_sync(__activemask(), val, 4);
    val += __shfl_down_sync(__activemask(), val, 2);
    val += __shfl_down_sync(__activemask(), val, 1);
    if(threadIdx.x % 32 == 0) {
      atomicAdd(output, val);
    }
  } else {
    atomicAdd(output, val);
  }
}

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

  int32_t f2 = (io * 256 + ii);
  int32_t f = f2 / (inter2_dimension);
  int32_t i = f / (inter3_dimension);
  int32_t iinter = i;
  int32_t iA = i;
  if (i >= inter1_dimension)
    return;

  int32_t l = f % (inter3_dimension);
  int32_t lA = iA * pack.ldA + l;
  if (l >= inter3_dimension)
    return;

  int32_t j = f2 % (inter2_dimension);
  int32_t jinter = iinter * inter2_dimension + j;
  int32_t linter = jinter * inter3_dimension + l;
  int32_t jC = j;
  int32_t lC = jC * pack.ldC + l;
  if (j >= inter2_dimension)
    return;

  atomicAddWarp(&A[lA], lA, inter[linter] * C[lC]);
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

  // Allocate an intermediate result T(i, j, l). Importantly, this call into the runtime must happen before
  // any CUDA calls. Otherwise, this can lead to a race where the task gets swapped out and the CUDA hijack
  // gets confused about what stream to send tasks on.
  T initVal = 0;
  Legion::DeferredBuffer<T, 1> buf(Legion::Memory::Kind::GPU_FB_MEM, Legion::DomainT<1>(Legion::Rect<1>(0, B1_dimension * C1_dimension * D2_dimension - 1)), &initVal);
  T* inter = buf.ptr(0);

  double alpha = 1.0000000000000000;
  cublasHandle_t handle = getCuBLAS();
  cudaStream_t taskStream = cudaStream_t();
  cudaStreamCreate(&(taskStream));
  CHECK_CUBLAS(cublasSetStream(handle, taskStream));

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
  contractInter<T><<<(B1_dimension * C1_dimension * D2_dimension + 255) / 256, 256, 0, taskStream>>>(pack, A_vals, C_vals, inter);
}

template<typename T, int DIM>
__device__ __inline__
size_t flattenPoint(T accessor, Legion::Point<DIM> point) {
  size_t base = 0;
  for (int i = 0; i < DIM; i++) {
    base += accessor.accessor.strides[i] * point[i];
  }
  return base;
}

// Hacky overload so that we can use atomic add warp with pointers and regions.
// However, we should be able to assert that the desired point when used with a
// pointer is 0.
template<typename T>
__device__ __inline__
size_t flattenPoint(T* pointer, int point) {
  return point;
}

template<typename T>
__global__
void ttv_kernel(int32_t iDim, int32_t jDim, int32_t kDim, size_t ldA, size_t ldB2, size_t ldB3, T* A_vals, const T* B_vals, const T* C_vals) {
  int32_t io = blockIdx.x;
  int32_t ii = (threadIdx.x % (64));
  if (threadIdx.x >= 64) {
    return;
  }

  int32_t f2 = io * 64 + ii;
  int32_t f = f2 / kDim;
  int32_t il = f / jDim;
  int32_t i = il;
  if (i >= iDim)
    return;

  int32_t jl = f % jDim;
  int32_t j = jl;
  size_t jB = i * ldB2 + j;
  int32_t jA = i * ldA + j;
  if (j >= jDim)
    return;

  int32_t k = f2 % kDim;
  size_t kB = jB * ldB3 + k;
  if (k >= kDim)
    return;

  atomicAddWarp(&A_vals[jA], jA, B_vals[kB] * C_vals[k]);
}

template <typename T>
void cu_ttv(TTVPack pack, T* A_vals, const T* B_vals, const T* C_vals) {
  auto iDim = pack.iDim;
  auto jDim = pack.jDim;
  auto kDim = pack.kDim;
  auto ldA = pack.ldA;
  auto ldB2 = pack.ldB2;
  auto ldB3 = pack.ldB3;
  ttv_kernel<T><<<((iDim * jDim * kDim) + 63) / 64, 64>>>(iDim, jDim, kDim, ldA, ldB2, ldB3, A_vals, B_vals, C_vals);
}

#endif // TACO_LG_CU_LEAF_KERNELS_H
