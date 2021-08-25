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

  int32_t block = blockIdx.x + ((0 / pieces) / 2048);
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);
  if (threadIdx.x >= 256) {
    return;
  }

  double precomputed[8 - 0];
  for (int32_t pprecomputed = 0; pprecomputed < (8 - 0); pprecomputed++) {
    precomputed[pprecomputed] = 0.0;
  }

  #pragma unroll 8
  for (int32_t thread_nz_pre = 0; thread_nz_pre < 8; thread_nz_pre++) {
    int32_t thread_nz = thread_nz_pre;
    int32_t fpos2 = (thread * 8 + thread_nz) + 0;
    int32_t fpos1 = (warp * 256 + fpos2) + 0;
    int32_t fposL = (block * 2048 + fpos1) + 0 / pieces;
    int32_t fposA = fposN * (((nnz) + (pieces - 1)) / pieces) + fposL;
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
  int32_t fpos2 = (thread * 8 + thread_nz) + 0;
  int32_t fpos1 = (warp * 256 + fpos2) + 0;
  int32_t fposL = (block * 2048 + fpos1) + 0 / pieces;
  int32_t fposA = fposN * (((nnz - 0) + (pieces - 1)) / pieces) + fposL;
  int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
  int32_t i = i_pos;
  for (int32_t thread_nz = 0; thread_nz < 8; thread_nz++) {
    int32_t fpos2 = (thread * 8 + thread_nz) + 0;
    int32_t fpos1 = (warp * 256 + fpos2) + 0;
    int32_t fposL = (block * 2048 + fpos1) + 0 / pieces;
    int32_t fposA = fposN * (((nnz) + (pieces - 1)) / pieces) + fposL;
    if (fposA >= (fposN + 1) * ((nnz + (pieces - 1)) / pieces - 0 / pieces))
      break;

    if (fposA >= nnz)
      break;

    int32_t f = A2_crd[fposA];
    while (fposA == A2_pos[i_pos].hi + 1) {
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
  auto numBlocks = (size + 2047) / 2048;

  AccessorRD y_vals(regions[0], FID_VALUE, LEGION_REDOP_SUM_FLOAT64);
  AccessorR A2_pos(regions[1], FID_RECT_1);
  AccessorI A2_crd(regions[2], FID_INDEX);
  AccessorD A_vals(regions[3], FID_VALUE);
  AccessorD x_vals(regions[4], FID_VALUE);

  int32_t initVal = 0;
  DeferredBuffer<int32_t, 1> i_blockStartsBuf(Memory::Kind::GPU_FB_MEM, DomainT<1>(Rect<1>(0, numBlocks)), &initVal);
  int32_t* i_blockStarts = i_blockStartsBuf.ptr(0);
  taco_binarySearchBeforeBlockLaunch(A2_pos, i_blockStarts, posDom.lo()[0], posDom.hi()[0], 2048, 256, numBlocks, nnzDom.lo()[0]);
//  int32_t* i_blockStarts = 0;
//  gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((((A2_pos[(1 * A1_dimension)] + (pieces - 1)) / pieces + 2047) / 2048 - (0 / pieces) / 2048) + 1)));
//  i_blockStarts = taco_binarySearchBeforeBlockLaunch(
//      A2_pos,
//      i_blockStarts,
//      0,
//      (1 * A1_dimension),
//      (0 * (((A2_pos[(1 * A1_dimension)] - 0) + (pieces - 1)) / pieces) + ((1 * 2048 + ((0 * 256 + ((0 * 8 + 0) + 0)) + 0)) + 0 / pieces)),
//      (((256 + 7) / 8 - 0 / 8) * ((2048 + 255) / 256 - 0 / 256)),
//      (((A2_pos[(1 * A1_dimension)] + (pieces - 1)) / pieces + 2047) / 2048 - (0 / pieces) / 2048)
//  );

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

//  for (int32_t fposN = 0; fposN < pieces; fposN++) {
//    if (((((A2_pos[A1_dimension] + (pieces - 1)) / pieces + 2047) / 2048 - (0 / pieces) / 2048)) > 0) {
  computeDeviceKernel0<<<numBlocks, (32 * 8)>>>(ar, i_blockStarts);
//    }
//  }



//  auto args = (spmvGPUArgs*)(task->args);
//  size_t A1_dimension = args->A1_dimension;
//  auto numBlocks = (args->nnz + 3583) / 3584;
//
//  AccessorWD y_vals(regions[0], FID_VALUE);
//  AccessorR A2_pos(regions[1], FID_RECT_1);
//  AccessorI A2_crd(regions[2], FID_INDEX);
//  AccessorD A_vals(regions[3], FID_VALUE);
//  AccessorD x_vals(regions[4], FID_VALUE);
//
//  int32_t initVal = 0;
//  DeferredBuffer<int32_t, 1> i_blockStartsBuf(Memory::Kind::GPU_FB_MEM, DomainT<1>(Rect<1>(0, numBlocks)), &initVal);
//  // DeferredBuffer<int32_t, 1> i_blockStartsBuf(Memory::Kind::Z_COPY_MEM, DomainT<1>(Rect<1>(0, numBlocks)), &initVal);
//  int32_t* i_blockStarts = i_blockStartsBuf.ptr(0);
//  taco_binarySearchBeforeBlockLaunch(A2_pos, i_blockStarts, (int32_t) 0, A1_dimension - 1, (int32_t) 3584, (int32_t) 512, numBlocks);
//
//  kernelArgs ar;
//  ar.A1_dimension = A1_dimension;
//  ar.nnz = args->nnz;
//  ar.y_vals = y_vals;
//  ar.A2_pos = A2_pos;
//  ar.A2_crd = A2_crd;
//  ar.A_vals = A_vals;
//  ar.x_vals = x_vals;
//
//  computeDeviceKernel0<<<numBlocks, (32 * 16)>>>(ar, i_blockStarts);
}

//void spmvgpu(Legion::Context ctx,
//             Legion::Runtime* runtime,
//             int32_t n, size_t nnz,
//             Legion::LogicalRegion y,
//             Legion::LogicalRegion A2_pos,
//             Legion::LogicalRegion A2_pos_par,
//             Legion::LogicalRegion A2_crd,
//             Legion::LogicalRegion A2_crd_par,
//             Legion::LogicalRegion A_vals,
//             Legion::LogicalRegion A_vals_par,
//             Legion::LogicalRegion x_vals) {
//  spmvGPUArgs args;
//  args.A1_dimension = n;
//  args.nnz = nnz;
//  TaskLauncher launcher(TID_SPMV_GPU, TaskArgument(&args, sizeof(spmvGPUArgs)));
//  launcher.add_region_requirement(RegionRequirement(y, READ_WRITE, EXCLUSIVE, y).add_field(FID_VALUE));
//  launcher.add_region_requirement(RegionRequirement(A2_pos, READ_ONLY, EXCLUSIVE, A2_pos_par, Mapping::DefaultMapper::EXACT_REGION).add_field(FID_RECT_1));
//  launcher.add_region_requirement(RegionRequirement(A2_crd, READ_ONLY, EXCLUSIVE, A2_crd_par, Mapping::DefaultMapper::EXACT_REGION).add_field(FID_INDEX));
//  launcher.add_region_requirement(RegionRequirement(A_vals, READ_ONLY, EXCLUSIVE, A_vals_par, Mapping::DefaultMapper::EXACT_REGION).add_field(FID_VALUE));
//  launcher.add_region_requirement(RegionRequirement(x_vals, READ_ONLY, EXCLUSIVE, x_vals).add_field(FID_VALUE));
//
//  // Run a few iterations of warmup.
//  runAsyncCall(ctx, runtime, [&]() {
//    for (int i = 0; i < 5; i++) {
//      runtime->execute_task(ctx, launcher);
//    }
//  });
//  std::vector<size_t> times;
//  benchmarkAsyncCall(ctx, runtime, times, [&]() {
//    for (int i = 0; i < 20; i++) {
//      runtime->execute_task(ctx, launcher);
//    }
//  });
//  LEGION_PRINT_ONCE(runtime, ctx, stdout, "Executed in %lf ms.\n", double(times[0]) / 20.0);
//}


void registerSPMVGPU() {
  {
    TaskVariantRegistrar registrar(TID_SPMV_POS_SPLIT, "spmvPos");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    Runtime::preregister_task_variant<spmvGPU>(registrar, "spmvPos");
  }
}
