#include "cublas_v2.h"
#include "cudalibs.h"
#include "leaf_kernels.cuh"
#include "taco_legion_header.h"
#include "taco_mapper.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
using namespace Legion;
typedef FieldAccessor<READ_ONLY,double,3,coord_t,Realm::AffineAccessor<double,3,coord_t>> AccessorROdouble3;

struct task_1Args {
  double a_val;
  int32_t b1_dimension;
  int32_t b2_dimension;
  int32_t b3_dimension;
};

LogicalPartition partition3Tensor(Context ctx, Runtime* runtime, LogicalRegion b, int32_t pieces) {
  int b1_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[0] + 1;
  int b2_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[1] + 1;
  int b3_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[2] + 1;
  auto b_index_space = get_index_space(b);

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto inIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(inIndexSpace));
  auto bDomain = runtime->get_index_space_domain(ctx, b_index_space);
  DomainPointColoring bColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    Point<3> bStart = Point<3>((in * ((b1_dimension + (pieces - 1)) / pieces) + 0 / pieces), 0, 0);
    Point<3> bEnd = Point<3>(TACO_MIN((in * ((b1_dimension + (pieces - 1)) / pieces) + ((b1_dimension + (pieces - 1)) / pieces - 1)),bDomain.hi()[0]), TACO_MIN(b2_dimension,bDomain.hi()[1]), TACO_MIN(b3_dimension,bDomain.hi()[2]));
    Rect<3> bRect = Rect<3>(bStart, bEnd);
    if (!bDomain.contains(bRect.lo) || !bDomain.contains(bRect.hi)) {
      bRect = bRect.make_empty();
    }
    bColoring[(*itr)] = bRect;
  }
  auto bPartition = runtime->create_index_partition(ctx, b_index_space, domain, bColoring, LEGION_DISJOINT_COMPLETE_KIND);
  return runtime->get_logical_partition(ctx, get_logical_region(b), bPartition);
}

__global__
void task_1DeviceKernel0(int64_t bPartitionBounds0hi, int64_t bPartitionBounds0lo, double* bufPtr, AccessorROdouble3 b_vals, AccessorROdouble3 c_vals, double a_val, int32_t b1_dimension, int32_t b2_dimension, int32_t b3_dimension, int32_t in) {

  int32_t io = blockIdx.x;
  int32_t ii = (threadIdx.x % (64));
  if (threadIdx.x >= 64) {
    return;
  }

  double tiia_val = 0.0;
  int32_t f2 = io * 64 + ii;
  int32_t f = f2 / b3_dimension;
  int32_t il = f / b2_dimension;
  int32_t i = il + bPartitionBounds0lo;
  if (i >= b1_dimension)
    return;

  if (i > bPartitionBounds0hi)
    return;

  int32_t j = f % b2_dimension;
  int32_t jb = i * 10 + j;
  int32_t jc = i * 10 + j;
  if (j >= b2_dimension)
    return;

  int32_t k = f2 % b3_dimension;
  Point<3> b_access_point = Point<3>(i, j, k);
  Point<3> c_access_point = Point<3>(i, j, k);
  if (k >= b3_dimension)
    return;

  tiia_val = tiia_val + b_vals[b_access_point] * c_vals[c_access_point];
  atomicAddWarp(&bufPtr[0], flattenPoint(bufPtr, 0), tiia_val);
}

double task_1(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion b = regions[0];
  PhysicalRegion c = regions[1];

  int32_t in = task->index_point[0];
  task_1Args* args = (task_1Args*)(task->args);
  double a_val = args->a_val;
  int32_t b1_dimension = args->b1_dimension;
  int32_t b2_dimension = args->b2_dimension;
  int32_t b3_dimension = args->b3_dimension;

  auto b_index_space = get_index_space(b);
  AccessorROdouble3 b_vals(b, FID_VAL);
  AccessorROdouble3 c_vals(c, FID_VAL);

  auto bPartitionBounds = runtime->get_index_space_domain(ctx, b_index_space);
  int64_t bPartitionBounds0lo = bPartitionBounds.lo()[0];
  int64_t bPartitionBounds0hi = bPartitionBounds.hi()[0];
  double init = 0;
  Legion::DeferredBuffer<double, 1> buf = Legion::DeferredBuffer<double, 1>(Legion::Memory::Kind::GPU_FB_MEM, DomainT<1>(Rect<1>(0, 0)), &(init));
  double* bufPtr = buf.ptr(0);

  task_1DeviceKernel0<<<((((bPartitionBounds0hi - bPartitionBounds0lo) + 1) * b2_dimension) * b3_dimension + 63) / 64, 64>>>(bPartitionBounds0hi, bPartitionBounds0lo, bufPtr, b_vals, c_vals, a_val, b1_dimension, b2_dimension, b3_dimension, in);

  cudaMemcpy(bufPtr, &(a_val), sizeof(a_val), cudaMemcpyHostToDevice);
  return a_val;
}

double computeLegion(Context ctx, Runtime* runtime, LogicalRegion b, LogicalRegion c, LogicalPartition bPartition, LogicalPartition cPartition) {
  int b1_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[0] + 1;
  int b2_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[1] + 1;
  int b3_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[2] + 1;

  double a_val = 0.0;

  DomainT<1> domain = runtime->get_index_partition_color_space(ctx, get_index_partition(bPartition));
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    DomainPoint domPoint = (*itr);
    auto bPartitionBounds = runtime->get_index_space_domain(runtime->get_logical_subregion_by_color(ctx, bPartition, domPoint).get_index_space());
  }
  RegionRequirement bReq = RegionRequirement(bPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(b));
  bReq.add_field(FID_VAL);
  RegionRequirement cReq = RegionRequirement(cPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(c));
  cReq.add_field(FID_VAL);
  task_1Args taskArgsRaw;
  taskArgsRaw.a_val = a_val;
  taskArgsRaw.b1_dimension = b1_dimension;
  taskArgsRaw.b2_dimension = b2_dimension;
  taskArgsRaw.b3_dimension = b3_dimension;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_1Args));
  IndexLauncher launcher = IndexLauncher(taskID(1), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(bReq);
  launcher.add_region_requirement(cReq);
  launcher.tag = launcher.tag | TACOMapper::UNTRACK_VALID_REGIONS;
  auto fm = runtime->execute_index_space(ctx, launcher);
  auto reduced = runtime->reduce_future_map(ctx, fm, LEGION_REDOP_SUM_FLOAT64);
  a_val = reduced.get<double>();


  return a_val;
}
void registerTacoTasks() {
  {
    TaskVariantRegistrar registrar(taskID(1), "task_1");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<double,task_1>(registrar, "task_1");
  }
}
