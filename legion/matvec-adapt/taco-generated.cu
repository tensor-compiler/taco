#include "cublas_v2.h"
#include "cudalibs.h"
#include "leaf_kernels.cuh"
#include "taco_legion_header.h"
#include "taco_mapper.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
using namespace Legion;
typedef FieldAccessor<READ_ONLY,double,1,coord_t,Realm::AffineAccessor<double,1,coord_t>> AccessorROdouble1;
typedef FieldAccessor<READ_ONLY,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorROdouble2;
typedef FieldAccessor<READ_WRITE,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorRWdouble2;

struct task_1Args {
  int32_t B1_dimension;
  int32_t B2_dimension;
};
struct task_2Args {
  int32_t B1_dimension;
  int32_t B2_dimension;
};

LogicalPartition partitionRows(Context ctx, Runtime* runtime, LogicalRegion A, int32_t pieces) {
  int A1_dimension = runtime->get_index_space_domain(get_index_space(A)).hi()[0] + 1;
  int A2_dimension = runtime->get_index_space_domain(get_index_space(A)).hi()[1] + 1;
  auto A_index_space = get_index_space(A);

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto inIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(inIndexSpace));
  auto ADomain = runtime->get_index_space_domain(ctx, A_index_space);
  DomainPointColoring AColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    Point<2> AStart = Point<2>((in * ((A1_dimension + (pieces - 1)) / pieces) + 0 / pieces), 0);
    Point<2> AEnd = Point<2>(TACO_MIN((in * ((A1_dimension + (pieces - 1)) / pieces) + ((A1_dimension + (pieces - 1)) / pieces - 1)),ADomain.hi()[0]), TACO_MIN(A2_dimension,ADomain.hi()[1]));
    Rect<2> ARect = Rect<2>(AStart, AEnd);
    if (!ADomain.contains(ARect.lo) || !ADomain.contains(ARect.hi)) {
      ARect = ARect.make_empty();
    }
    AColoring[(*itr)] = ARect;
  }
  auto APartition = runtime->create_index_partition(ctx, A_index_space, domain, AColoring, LEGION_DISJOINT_COMPLETE_KIND);
  return runtime->get_logical_partition(ctx, get_logical_region(A), APartition);
}

LogicalPartition partitionC(Context ctx, Runtime* runtime, LogicalRegion C, int32_t pieces) {
  int C1_dimension = runtime->get_index_space_domain(get_index_space(C)).hi()[0] + 1;
  auto C_index_space = get_index_space(C);

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto inIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(inIndexSpace));
  auto CDomain = runtime->get_index_space_domain(ctx, C_index_space);
  DomainPointColoring CColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    Point<1> CStart = Point<1>((in * ((C1_dimension + (pieces - 1)) / pieces) + 0 / pieces));
    Point<1> CEnd = Point<1>(TACO_MIN((in * ((C1_dimension + (pieces - 1)) / pieces) + ((C1_dimension + (pieces - 1)) / pieces - 1)),CDomain.hi()[0]));
    Rect<1> CRect = Rect<1>(CStart, CEnd);
    if (!CDomain.contains(CRect.lo) || !CDomain.contains(CRect.hi)) {
      CRect = CRect.make_empty();
    }
    CColoring[(*itr)] = CRect;
  }
  auto CPartition = runtime->create_index_partition(ctx, C_index_space, domain, CColoring, LEGION_DISJOINT_COMPLETE_KIND);
  return runtime->get_logical_partition(ctx, get_logical_region(C), CPartition);
}

__global__
void task_1DeviceKernel0(int64_t APartitionBounds0hi, int64_t APartitionBounds0lo, AccessorRWdouble2 A_vals, AccessorROdouble2 B_vals, AccessorROdouble1 C_vals, int32_t B1_dimension, int32_t B2_dimension, int32_t in) {

  int32_t io = blockIdx.x;
  int32_t ii = (threadIdx.x % (64));
  if (threadIdx.x >= 64) {
    return;
  }

  int32_t f = io * 64 + ii;
  int32_t il = f / B2_dimension;
  int32_t i = il + APartitionBounds0lo;
  if (i >= B1_dimension)
    return;

  if (i > APartitionBounds0hi)
    return;

  int32_t j = f % B2_dimension;
  Point<2> A_access_point = Point<2>(i, j);
  Point<2> B_access_point = Point<2>(i, j);
  Point<1> C_access_point = Point<1>(j);
  if (j >= B2_dimension)
    return;

  A_vals[A_access_point] = B_vals[B_access_point] * C_vals[C_access_point];
}

void task_1(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion A = regions[0];
  PhysicalRegion B = regions[1];
  PhysicalRegion C = regions[2];

  int32_t in = task->index_point[0];
  task_1Args* args = (task_1Args*)(task->args);
  int32_t B1_dimension = args->B1_dimension;
  int32_t B2_dimension = args->B2_dimension;

  auto A_index_space = get_index_space(A);
  AccessorROdouble1 C_vals(C, FID_VAL);
  AccessorROdouble2 B_vals(B, FID_VAL);
  AccessorRWdouble2 A_vals(A, FID_VAL);

  auto APartitionBounds = runtime->get_index_space_domain(ctx, A_index_space);
  int64_t APartitionBounds0lo = APartitionBounds.lo()[0];
  int64_t APartitionBounds0hi = APartitionBounds.hi()[0];
  if ((((((APartitionBounds0hi - APartitionBounds0lo) + 1) * B2_dimension + 63) / 64)) > 0) {
    task_1DeviceKernel0<<<((((APartitionBounds0hi - APartitionBounds0lo) + 1) * B2_dimension + 63) / 64), 64>>>(APartitionBounds0hi, APartitionBounds0lo, A_vals, B_vals, C_vals, B1_dimension, B2_dimension, in);
  }}

void computeLegionRows(Context ctx, Runtime* runtime, LogicalRegion A, LogicalRegion B, LogicalRegion C, LogicalPartition APartition, LogicalPartition BPartition) {
  int B1_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[0] + 1;
  int B2_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[1] + 1;
  auto C_index_space = get_index_space(C);

  DomainT<1> domain = runtime->get_index_partition_color_space(ctx, get_index_partition(APartition));
  auto CDomain = runtime->get_index_space_domain(ctx, C_index_space);
  DomainPointColoring CColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    DomainPoint domPoint = (*itr);
    auto APartitionBounds = runtime->get_index_space_domain(runtime->get_logical_subregion_by_color(ctx, APartition, domPoint).get_index_space());
  }
  RegionRequirement AReq = RegionRequirement(APartition, 0, READ_WRITE, EXCLUSIVE, get_logical_region(A));
  AReq.add_field(FID_VAL);
  RegionRequirement BReq = RegionRequirement(BPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(B));
  BReq.add_field(FID_VAL);
  RegionRequirement CReq = RegionRequirement(get_logical_region(C), READ_ONLY, EXCLUSIVE, get_logical_region(C));
  CReq.add_field(FID_VAL);
  task_1Args taskArgsRaw;
  taskArgsRaw.B1_dimension = B1_dimension;
  taskArgsRaw.B2_dimension = B2_dimension;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_1Args));
  IndexLauncher launcher = IndexLauncher(taskID(1), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(AReq);
  launcher.add_region_requirement(BReq);
  launcher.add_region_requirement(CReq);
  launcher.tag = launcher.tag | TACOMapper::UNTRACK_VALID_REGIONS;
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();

}

__global__
void task_2DeviceKernel0(int64_t APartitionBounds1hi, int64_t APartitionBounds1lo, AccessorRWdouble2 A_vals, AccessorROdouble2 B_vals, AccessorROdouble1 C_vals, int32_t B1_dimension, int32_t B2_dimension, int32_t jn) {

  int32_t io = blockIdx.x + (0 / 64);
  int32_t ii = (threadIdx.x % (64));
  if (threadIdx.x >= 64) {
    return;
  }

  int32_t f = io * 64 + ii;
  int32_t i = f / ((APartitionBounds1hi - APartitionBounds1lo) + 1);
  if (i >= B1_dimension)
    return;

  int32_t jl = f % ((APartitionBounds1hi - APartitionBounds1lo) + 1);
  int32_t j = jl + APartitionBounds1lo;
  Point<2> A_access_point = Point<2>(i, j);
  Point<2> B_access_point = Point<2>(i, j);
  Point<1> C_access_point = Point<1>(j);
  if (j >= B2_dimension)
    return;

  if (j > APartitionBounds1hi)
    return;

  A_vals[A_access_point] = B_vals[B_access_point] * C_vals[C_access_point];
}

void task_2(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion A = regions[0];
  PhysicalRegion B = regions[1];
  PhysicalRegion C = regions[2];

  int32_t jn = task->index_point[0];
  task_2Args* args = (task_2Args*)(task->args);
  int32_t B1_dimension = args->B1_dimension;
  int32_t B2_dimension = args->B2_dimension;

  auto A_index_space = get_index_space(A);
  AccessorROdouble1 C_vals(C, FID_VAL);
  AccessorROdouble2 B_vals(B, FID_VAL);
  AccessorRWdouble2 A_vals(A, FID_VAL);

  auto APartitionBounds = runtime->get_index_space_domain(ctx, A_index_space);
  int64_t APartitionBounds1lo = APartitionBounds.lo()[1];
  int64_t APartitionBounds1hi = APartitionBounds.hi()[1];
  if ((((B1_dimension * ((APartitionBounds1hi - APartitionBounds1lo) + 1) + 63) / 64 - 0 / 64)) > 0) {
    task_2DeviceKernel0<<<((B1_dimension * ((APartitionBounds1hi - APartitionBounds1lo) + 1) + 63) / 64 - 0 / 64), 64>>>(APartitionBounds1hi, APartitionBounds1lo, A_vals, B_vals, C_vals, B1_dimension, B2_dimension, jn);
  }}

void computeLegionCols(Context ctx, Runtime* runtime, LogicalRegion A, LogicalRegion B, LogicalRegion C, LogicalPartition APartition, LogicalPartition BPartition, LogicalPartition CPartition) {
  int B1_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[0] + 1;
  int B2_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[1] + 1;

  DomainT<1> domain = runtime->get_index_partition_color_space(ctx, get_index_partition(APartition));
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    DomainPoint domPoint = (*itr);
    auto APartitionBounds = runtime->get_index_space_domain(runtime->get_logical_subregion_by_color(ctx, APartition, domPoint).get_index_space());
  }
  RegionRequirement AReq = RegionRequirement(APartition, 0, READ_WRITE, EXCLUSIVE, get_logical_region(A));
  AReq.add_field(FID_VAL);
  RegionRequirement BReq = RegionRequirement(BPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(B));
  BReq.add_field(FID_VAL);
  RegionRequirement CReq = RegionRequirement(CPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(C));
  CReq.add_field(FID_VAL);
  task_2Args taskArgsRaw;
  taskArgsRaw.B1_dimension = B1_dimension;
  taskArgsRaw.B2_dimension = B2_dimension;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_2Args));
  IndexLauncher launcher = IndexLauncher(taskID(2), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(AReq);
  launcher.add_region_requirement(BReq);
  launcher.add_region_requirement(CReq);
  launcher.tag = launcher.tag | TACOMapper::UNTRACK_VALID_REGIONS;
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();

}
void registerTacoTasks() {
  {
    TaskVariantRegistrar registrar(taskID(1), "task_1");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_1>(registrar, "task_1");
  }
  {
    TaskVariantRegistrar registrar(taskID(2), "task_2");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_2>(registrar, "task_2");
  }
}
