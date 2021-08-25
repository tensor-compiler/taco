#inclue "legion.h"
#include "handwritten.h"

using namespace Legion;

typedef FieldAccessor<READ_ONLY,int32_t,1,coord_t,Realm::AffineAccessor<int32_t,1,coord_t>> AccessorI;
typedef FieldAccessor<READ_ONLY,Rect<1>,1,coord_t,Realm::AffineAccessor<Rect<1>,1,coord_t>> AccessorR;
typedef FieldAccessor<READ_ONLY,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorD;
typedef FieldAccessor<READ_WRITE,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorWD;

struct args {
  int32_t A1_dimension;
  size_t nnz;
  AccessorD y_vals;
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

__device__ __host__ int taco_binarySearchBefore(AccessorR *array, int arrayStart, int arrayEnd, int target) {
  if (array[arrayEnd] <= target) {
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
__global__ void taco_binarySearchBeforeBlock(AccessorR array, int * results, int arrayStart, int arrayEnd, int values_per_block, int num_blocks) {
  int thread = threadIdx.x;
  int block = blockIdx.x;
  int idx = block * blockDim.x + thread;
  if (idx >= num_blocks+1) {
    return;
  }

  results[idx] = taco_binarySearchBefore(array, arrayStart, arrayEnd, idx * values_per_block);
}

__host__ int * taco_binarySearchBeforeBlockLaunch(AccessorR array, int * results, int arrayStart, int arrayEnd, int values_per_block, int block_size, int num_blocks){
  int num_search_blocks = (num_blocks + 1 + block_size - 1) / block_size;
  taco_binarySearchBeforeBlock<<<num_search_blocks, block_size>>>(array, results, arrayStart, arrayEnd, values_per_block, num_blocks);
  return results;
}

__global__
void computeDeviceKernel0(args ar, int32_t* i_blockStarts){
  int A1_dimension = ar.A1_dimension;
  int nnz = ar.nnz;
  auto A2_pos = ar.A2_pos;
  auto A2_crd = ar.A2_crd;
  auto A_vals = ar.A_vals;
  auto x_vals = ar.x_vals;
  auto y_vals = ar.y_vals;

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);
  if (threadIdx.x >= 512) {
    return;
  }

  double workspace[7];
  for (int32_t pworkspace = 0; pworkspace < 7; pworkspace++) {
    workspace[pworkspace] = 0.0;
  }
  int32_t thr_nz = 0;
  int32_t fpos2 = thread * 7 + thr_nz;
  int32_t fpos1 = warp * 224 + fpos2;
  int32_t fposA = block * 3584 + fpos1;
  int32_t f = A2_crd[fposA];
  if ((block * 3584 + fpos1) + 7 >= nnz) {
    for (int32_t thr_nz_pre = 0; thr_nz_pre < 7; thr_nz_pre++) {
      int32_t thr_nz = thr_nz_pre;
      int32_t fpos2 = thread * 7 + thr_nz;
      int32_t fpos1 = warp * 224 + fpos2;
      int32_t fposA = block * 3584 + fpos1;
      if (fposA >= nnz)
        break;

      int32_t f = A2_crd[fposA];
      workspace[thr_nz_pre] = workspace[thr_nz_pre] + A_vals[fposA] * x_vals[f];
    }
  }
  else {
#pragma unroll 7
    for (int32_t thr_nz_pre = 0; thr_nz_pre < 7; thr_nz_pre++) {
      int32_t thr_nz = thr_nz_pre;
      int32_t fpos2 = thread * 7 + thr_nz;
      int32_t fpos1 = warp * 224 + fpos2;
      int32_t fposA = block * 3584 + fpos1;
      int32_t f = A2_crd[fposA];
      workspace[thr_nz_pre] = workspace[thr_nz_pre] + A_vals[fposA] * x_vals[f];
    }
  }
  int32_t pA2_begin = i_blockStarts[block];
  int32_t pA2_end = i_blockStarts[(block + 1)];
  int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
  int32_t i = i_pos;
  for (int32_t thr_nz = 0; thr_nz < 7; thr_nz++) {
    int32_t fpos2 = thread * 7 + thr_nz;
    int32_t fpos1 = warp * 224 + fpos2;
    int32_t fposA = block * 3584 + fpos1;
    if (fposA >= nnz)
      break;

    int32_t f = A2_crd[fposA];
    while (fposA == A2_pos[i_pos].hi + 1) {
      i_pos = i_pos + 1;
      i = i_pos;
    }
    atomicAddWarp(&y_vals[i], i, workspace[thr_nz]);
  }
}

struct spmvGPUArgs {
  size_t nnz;
  size_t A1_dimension;
};

void spmvGPU(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  auto args = (spmvGPUArgs*)(task->args);
  size_t A1_dimension = args->A1_dimension;
  auto numBlocks = (nnz + 3583) / 3584;

  AccessorD y_vals(regions[0], FID_VALUE);
  AccessorR A2_pos(regions[1], FID_RECT_1);
  AccessorI A2_crd(regions[2], FID_INDEX);
  AccessorD A_vals(regions[3], FID_VALUE);
  AccessorD x_vals(regions[4], FID_VALUE);

  double initVal = 0;
  DeferredBuffer<int32_t, 1> i_blockStartsBuf(Memory::Kind::GPU_FB_MEM, DomainT<1>(Rect<1>(0, numBlocks)));
  int32_t i_blockStarts = i_blockStartsBuf.ptr(0);
  taco_binarySearchBeforeBlockLaunch(A2_pos, i_blockStarts, (int32_t) 0, A1_dimension, (int32_t) 3584, (int32_t) 512, numBlocks);

  args ar;
  ar.A1_dimension = A1_dimension;
  ar.nnz = args->nnz;
  ar.y_vals = y_vals;
  ar.A2_pos = A2_pos;
  ar.A2_crd = A2_crd;
  ar.A_vals = A_vals;
  ar.x_vals = x_vals;

  computeDeviceKernel0<<<(numBlocks, (32 * 16)>>>(ar, i_blockStarts);
}

void spmvgpu(Legion::Context ctx,
             Legion::Runtime* runtime,
             int32_t n, size_t nnz,
             Legion::LogicalRegion y,
             Legion::LogicalRegion A2_pos,
             Legion::LogicalRegion A2_pos_par,
             Legion::LogicalRegion A2_crd,
             Legion::LogicalRegion A2_crd_par,
             Legion::LogicalRegion A_vals,
             Legion::LogicalRegion A_vals_par,
             Legion::LogicalRegion x_vals) {
  spmvGPUArgs args;
  args.A1_dimension = n;
  args.nnz = nnz;
  TaskLauncher launcher(TID_SPMV_GPU, TaskArgument(&args, sizeof(spmvGPUArgs)));
  launcher.add_region_requirement(RegionRequirement(y, READ_WRITE, EXCLUSIVE, y).add_field(FID_VALUE));
  launcher.add_region_requirement(RegionRequirement(A2_pos, READ_ONLY, EXCLUSIVE, A2_pos).add_field(FID_RECT_1));
  launcher.add_region_requirement(RegionRequirement(A2_crd, READ_ONLY, EXCLUSIVE, A2_crd).add_field(FID_INDEX));
  launcher.add_region_requirement(RegionRequirement(A_vals, READ_ONLY, EXCLUSIVE, A_vals).add_field(FID_INDEX));
  launcher.add_region_requirement(RegionRequirement(x_vals, READ_ONLY, EXCLUSIVE, x_vals).add_field(FID_VALUE));
  runtime->execute_task(ctx, launcher).wait();
}


void registerSPMVGPU() {
  {
    TaskVariantRegistrar registrar(TID_SPMV_GPU, "spmvGPU");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    Runtime::preregister_task_variant<spmvGPU>(registrar, "spmvGPU");
  }
}