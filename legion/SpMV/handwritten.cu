#include "legion.h"
#include "mappers/default_mapper.h"
#include "handwritten.h"

using namespace Legion;

typedef ReductionAccessor<SumReduction<double>,true,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorRD;
typedef FieldAccessor<READ_ONLY,int32_t,1,coord_t,Realm::AffineAccessor<int32_t,1,coord_t>> AccessorI;
typedef FieldAccessor<READ_ONLY,Rect<1>,1,coord_t,Realm::AffineAccessor<Rect<1>,1,coord_t>> AccessorR;
typedef FieldAccessor<READ_ONLY,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorD;
typedef FieldAccessor<READ_WRITE,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorWD;

struct kernelArgs {
  int32_t A1_dimension;
  int fposN;
  size_t nnz;
  int pieces;
  // AccessorWD y_vals;
  AccessorRD y_vals;
  AccessorD x_vals;
  AccessorR A2_pos;
  AccessorI A2_crd;
  AccessorD A_vals;
};

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

__device__ __host__ int taco_binarySearchBefore(AccessorR array, int arrayStart, int arrayEnd, int target) {
  if (array[arrayEnd].hi <= target) {
    return arrayEnd;
  }
  int lowerBound = arrayStart; // always <= target
  int upperBound = arrayEnd; // always > target
  while (upperBound - lowerBound > 1) {
    int mid = (upperBound + lowerBound) / 2;
    Rect<1> midValue = array[mid];
    if (midValue.hi < target) {
      lowerBound = mid;
    }
    else if (midValue.lo > target) {
      upperBound = mid;
    }
    else {
      return mid;
    }
  }
  return lowerBound;
}
__global__ void taco_binarySearchBeforeBlock(AccessorR array, int * results, int arrayStart, int arrayEnd, int values_per_block, int num_blocks, int offset) {
  int thread = threadIdx.x;
  int block = blockIdx.x;
  int idx = block * blockDim.x + thread;
  if (idx >= num_blocks+1) {
    return;
  }
  results[idx] = taco_binarySearchBefore(array, arrayStart, arrayEnd, idx * values_per_block + offset);
}

__host__ int * taco_binarySearchBeforeBlockLaunch(AccessorR array, int * results, int arrayStart, int arrayEnd, int values_per_block, int block_size, int num_blocks, int offset){
  int num_search_blocks = (num_blocks + 1 + block_size - 1) / block_size;
  taco_binarySearchBeforeBlock<<<num_search_blocks, block_size>>>(array, results, arrayStart, arrayEnd, values_per_block, num_blocks, offset);
  return results;
}

__global__
void computeDeviceKernel0(kernelArgs ar, int32_t* i_blockStarts){
  int nnz = ar.nnz;
  int fposN = ar.fposN;
  int pieces = ar.pieces;
  auto A2_pos = ar.A2_pos;
  auto A2_crd = ar.A2_crd;
  auto A_vals = ar.A_vals;
  auto x_vals = ar.x_vals;
  auto y_vals = ar.y_vals;

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);
  if (threadIdx.x >= 3584) {
    return;
  }

  double precomputed[7];
  for (int32_t pprecomputed = 0; pprecomputed < (7 - 0); pprecomputed++) {
    precomputed[pprecomputed] = 0.0;
  }

  #pragma unroll 7
  for (int32_t thread_nz_pre = 0; thread_nz_pre < 7; thread_nz_pre++) {
    int32_t thread_nz = thread_nz_pre;
    int32_t fpos2 = (thread * 7 + thread_nz) + 0;
    int32_t fpos1 = (warp * 224 + fpos2) + 0;
    int32_t fposL = (block * 3584 + fpos1) + 0 / pieces;
    int32_t fposA = fposN * ((nnz + (pieces - 1)) / pieces) + fposL;
    if (fposA >= (fposN + 1) * ((nnz + (pieces - 1)) / pieces - 0 / pieces))
      break;

    if (fposA >= nnz)
      break;

    int32_t f = A2_crd[fposA];
    int32_t j = f;
    int32_t jx = j;
    precomputed[thread_nz_pre] = A_vals[fposA] * x_vals[jx];
  }
  int32_t pA2_begin = i_blockStarts[block];
  int32_t pA2_end = i_blockStarts[(block + 1)];
  int32_t thread_nz = 0;
  int32_t fpos2 = (thread * 7 + thread_nz) + 0;
  int32_t fpos1 = (warp * 224 + fpos2) + 0;
  int32_t fposL = (block * 3584 + fpos1) + 0 / pieces;
  int32_t fposA = fposN * (((nnz - 0) + (pieces - 1)) / pieces) + fposL;
  int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
  int32_t i = i_pos;
  for (int32_t thread_nz = 0; thread_nz < 7; thread_nz++) {
    int32_t fpos2 = (thread * 7 + thread_nz) + 0;
    int32_t fpos1 = (warp * 224 + fpos2) + 0;
    int32_t fposL = (block * 3584 + fpos1) + 0 / pieces;
    int32_t fposA = fposN * ((nnz + (pieces - 1)) / pieces) + fposL;
    if (fposA >= (fposN + 1) * ((nnz + (pieces - 1)) / pieces - 0 / pieces))
      break;

    if (fposA >= nnz)
      break;

    int32_t f = A2_crd[fposA];
    while (fposA == A2_pos[(i_pos)].hi + 1) {
      i_pos = i_pos + 1;
      i = i_pos;
    }
    int32_t iy = i;
    atomicAddWarp(y_vals.ptr(iy), iy, precomputed[thread_nz]);
  }
}

struct spmvGPUArgs {
  size_t nnz;
  size_t A1_dimension;
};

void spmvGPU(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  auto args = (spmvPosSplitArgs*)(task->args);
  int fposN = task->index_point[0];

  auto posDom = runtime->get_index_space_domain(regions[1].get_logical_region().get_index_space());
  auto nnzDom = runtime->get_index_space_domain(regions[3].get_logical_region().get_index_space());
  auto size = nnzDom.get_volume();
  auto numBlocks = (size + 3583) / 3584;

  AccessorRD y_vals(regions[0], FID_VALUE, LEGION_REDOP_SUM_FLOAT64);
  AccessorR A2_pos(regions[1], FID_RECT_1);
  AccessorI A2_crd(regions[2], FID_INDEX);
  AccessorD A_vals(regions[3], FID_VALUE);
  AccessorD x_vals(regions[4], FID_VALUE);

  // PERFORMANCE NOTE: Larger blocks are better. Doubling the block size led to a 2x performace improvement.

  int32_t initVal = 0;
  DeferredBuffer<int32_t, 1> i_blockStartsBuf(Memory::Kind::GPU_FB_MEM, DomainT<1>(Rect<1>(0, numBlocks)), &initVal);
  int32_t* i_blockStarts = i_blockStartsBuf.ptr(0);
  taco_binarySearchBeforeBlockLaunch(A2_pos, i_blockStarts, posDom.lo()[0], posDom.hi()[0], 3548, 512, numBlocks, nnzDom.lo()[0]);

  kernelArgs ar;
  ar.A1_dimension = args->A1_dimension;
  ar.nnz = args->nnz;
  ar.fposN = fposN;
  ar.pieces = args->pieces;
  ar.y_vals = y_vals;
  ar.A2_pos = A2_pos;
  ar.A2_crd = A2_crd;
  ar.A_vals = A_vals;
  ar.x_vals = x_vals;

  computeDeviceKernel0<<<numBlocks, (32 * 16)>>>(ar, i_blockStarts);
}

void registerSPMVGPU() {
  {
    TaskVariantRegistrar registrar(TID_SPMV_POS_SPLIT, "spmvPos");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<spmvGPU>(registrar, "spmvPos");
  }
}
