#include "cublas_v2.h"
#include "cudalibs.h"
#include "leaf_kernels.cuh"
#include "taco_legion_header.h"
#include "taco_mapper.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
using namespace Legion;
typedef FieldAccessor<READ_ONLY,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorROdouble2;
typedef FieldAccessor<READ_ONLY,double,3,coord_t,Realm::AffineAccessor<double,3,coord_t>> AccessorROdouble3;
typedef FieldAccessor<READ_WRITE,double,3,coord_t,Realm::AffineAccessor<double,3,coord_t>> AccessorRWdouble3;

struct task_1Args {
  int32_t pieces;
};
struct task_2Args {
  int32_t pieces;
};
struct task_3Args {
  int32_t pieces;
};
struct task_4Args {
};

LogicalPartition partition3Tensor(Context ctx, Runtime* runtime, LogicalRegion A, int32_t pieces) {
  int A1_dimension = runtime->get_index_space_domain(get_index_space(A)).hi()[0] + 1;
  int A2_dimension = runtime->get_index_space_domain(get_index_space(A)).hi()[1] + 1;
  int A3_dimension = runtime->get_index_space_domain(get_index_space(A)).hi()[2] + 1;
  auto A_index_space = get_index_space(A);

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto inIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(inIndexSpace));
  auto ADomain = runtime->get_index_space_domain(ctx, A_index_space);
  DomainPointColoring AColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    Point<3> AStart = Point<3>((in * ((A1_dimension + (pieces - 1)) / pieces) + 0 / pieces), 0, 0);
    Point<3> AEnd = Point<3>(TACO_MIN((in * ((A1_dimension + (pieces - 1)) / pieces) + ((A1_dimension + (pieces - 1)) / pieces - 1)),ADomain.hi()[0]), TACO_MIN(A2_dimension,ADomain.hi()[1]), TACO_MIN(A3_dimension,ADomain.hi()[2]));
    Rect<3> ARect = Rect<3>(AStart, AEnd);
    if (!ADomain.contains(ARect.lo) || !ADomain.contains(ARect.hi)) {
      ARect = ARect.make_empty();
    }
    AColoring[(*itr)] = ARect;
  }
  auto APartition = runtime->create_index_partition(ctx, A_index_space, domain, AColoring, LEGION_DISJOINT_COMPLETE_KIND);
  return runtime->get_logical_partition(ctx, get_logical_region(A), APartition);
}

void task_1(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion A = regions[0];

  int32_t in = task->index_point[0];
  task_1Args* args = (task_1Args*)(task->args);
  int32_t pieces = args->pieces;


}

LogicalPartition placeLegionA(Context ctx, Runtime* runtime, LogicalRegion A, int32_t pieces) {
  int A1_dimension = runtime->get_index_space_domain(get_index_space(A)).hi()[0] + 1;
  int A2_dimension = runtime->get_index_space_domain(get_index_space(A)).hi()[1] + 1;
  int A3_dimension = runtime->get_index_space_domain(get_index_space(A)).hi()[2] + 1;
  auto A_index_space = get_index_space(A);

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto inIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(inIndexSpace));
  auto ADomain = runtime->get_index_space_domain(ctx, A_index_space);
  DomainPointColoring AColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    Point<3> AStart = Point<3>((in * ((A1_dimension + (pieces - 1)) / pieces) + 0 / pieces), 0, 0);
    Point<3> AEnd = Point<3>(TACO_MIN((in * ((A1_dimension + (pieces - 1)) / pieces) + ((A1_dimension + (pieces - 1)) / pieces - 1)),ADomain.hi()[0]), TACO_MIN(A2_dimension,ADomain.hi()[1]), TACO_MIN(A3_dimension,ADomain.hi()[2]));
    Rect<3> ARect = Rect<3>(AStart, AEnd);
    if (!ADomain.contains(ARect.lo) || !ADomain.contains(ARect.hi)) {
      ARect = ARect.make_empty();
    }
    AColoring[(*itr)] = ARect;
  }
  auto APartition = runtime->create_index_partition(ctx, A_index_space, domain, AColoring, LEGION_COMPUTE_KIND);
  LogicalPartition ALogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(A), APartition);
  RegionRequirement AReq = RegionRequirement(ALogicalPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(A));
  AReq.add_field(FID_VAL);
  task_1Args taskArgsRaw;
  taskArgsRaw.pieces = pieces;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_1Args));
  IndexLauncher launcher = IndexLauncher(taskID(1), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(AReq);
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  return runtime->get_logical_partition(ctx, get_logical_region(A), APartition);

}

void task_2(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion B = regions[0];

  int32_t in = task->index_point[0];
  task_2Args* args = (task_2Args*)(task->args);
  int32_t pieces = args->pieces;


}

LogicalPartition placeLegionB(Context ctx, Runtime* runtime, LogicalRegion B, int32_t pieces) {
  int B1_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[0] + 1;
  int B2_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[1] + 1;
  int B3_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[2] + 1;
  auto B_index_space = get_index_space(B);

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto inIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(inIndexSpace));
  auto BDomain = runtime->get_index_space_domain(ctx, B_index_space);
  DomainPointColoring BColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    Point<3> BStart = Point<3>((in * ((B1_dimension + (pieces - 1)) / pieces) + 0 / pieces), 0, 0);
    Point<3> BEnd = Point<3>(TACO_MIN((in * ((B1_dimension + (pieces - 1)) / pieces) + ((B1_dimension + (pieces - 1)) / pieces - 1)),BDomain.hi()[0]), TACO_MIN(B2_dimension,BDomain.hi()[1]), TACO_MIN(B3_dimension,BDomain.hi()[2]));
    Rect<3> BRect = Rect<3>(BStart, BEnd);
    if (!BDomain.contains(BRect.lo) || !BDomain.contains(BRect.hi)) {
      BRect = BRect.make_empty();
    }
    BColoring[(*itr)] = BRect;
  }
  auto BPartition = runtime->create_index_partition(ctx, B_index_space, domain, BColoring, LEGION_COMPUTE_KIND);
  LogicalPartition BLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(B), BPartition);
  RegionRequirement BReq = RegionRequirement(BLogicalPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(B));
  BReq.add_field(FID_VAL);
  task_2Args taskArgsRaw;
  taskArgsRaw.pieces = pieces;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_2Args));
  IndexLauncher launcher = IndexLauncher(taskID(2), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(BReq);
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  return runtime->get_logical_partition(ctx, get_logical_region(B), BPartition);

}

void task_3(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion C = regions[0];

  int32_t kn = task->index_point[0];
  task_3Args* args = (task_3Args*)(task->args);
  int32_t pieces = args->pieces;


}

LogicalPartition placeLegionC(Context ctx, Runtime* runtime, LogicalRegion C, int32_t pieces) {
  int C1_dimension = runtime->get_index_space_domain(get_index_space(C)).hi()[0] + 1;
  int C2_dimension = runtime->get_index_space_domain(get_index_space(C)).hi()[1] + 1;
  auto C_index_space = get_index_space(C);

  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((pieces - 1));
  auto knIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(knIndexSpace));
  auto CDomain = runtime->get_index_space_domain(ctx, C_index_space);
  DomainPointColoring CColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    Point<2> CStart = Point<2>(0, 0);
    Point<2> CEnd = Point<2>(TACO_MIN(C1_dimension,CDomain.hi()[0]), TACO_MIN(C2_dimension,CDomain.hi()[1]));
    Rect<2> CRect = Rect<2>(CStart, CEnd);
    if (!CDomain.contains(CRect.lo) || !CDomain.contains(CRect.hi)) {
      CRect = CRect.make_empty();
    }
    CColoring[(*itr)] = CRect;
  }
  auto CPartition = runtime->create_index_partition(ctx, C_index_space, domain, CColoring, LEGION_COMPUTE_KIND);
  LogicalPartition CLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(C), CPartition);
  RegionRequirement CReq = RegionRequirement(CLogicalPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(C));
  CReq.add_field(FID_VAL);
  task_3Args taskArgsRaw;
  taskArgsRaw.pieces = pieces;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_3Args));
  IndexLauncher launcher = IndexLauncher(taskID(3), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(CReq);
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  return runtime->get_logical_partition(ctx, get_logical_region(C), CPartition);

}

void task_4(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion A = regions[0];
  PhysicalRegion B = regions[1];
  PhysicalRegion C = regions[2];

  int32_t in = task->index_point[0];
  auto A_index_space = get_index_space(A);
  auto B_index_space = get_index_space(B);
  auto C_index_space = get_index_space(C);
  AccessorROdouble2 C_vals(C, FID_VAL);
  AccessorROdouble3 B_vals(B, FID_VAL);
  AccessorRWdouble3 A_vals(A, FID_VAL);

  auto APartitionBounds = runtime->get_index_space_domain(ctx, A_index_space);
  auto aDomain = runtime->get_index_space_domain(ctx, A_index_space);
  auto bDomain = runtime->get_index_space_domain(ctx, B_index_space);
  auto cDomain = runtime->get_index_space_domain(ctx, C_index_space);
  if (bDomain.get_volume() == 0 || cDomain.get_volume() == 0)
    return ;

  double alpha = 1.0000000000000000;
  cublasHandle_t handle = getCuBLAS();
  cudaStream_t taskStream = cudaStream_t();
  cudaStreamCreate(&(taskStream));
  CHECK_CUBLAS(cublasSetStream(handle, taskStream));
  for (int32_t loopIdx = 0; loopIdx < (1 + (aDomain.hi()[0] - aDomain.lo()[0])); loopIdx++) {
    CHECK_CUBLAS(cublasDgemm(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      (1 + (cDomain.hi()[1] - cDomain.lo()[1])),
      (1 + (bDomain.hi()[1] - bDomain.lo()[1])),
      (1 + (cDomain.hi()[0] - cDomain.lo()[0])),
      &(alpha),
      C_vals.ptr(cDomain.lo()),
      (C_vals.accessor.strides[0] / sizeof(double)),
      (B_vals.ptr(bDomain.lo()) + (B_vals.accessor.strides[0] / sizeof(double)) * loopIdx),
      (B_vals.accessor.strides[1] / sizeof(double)),
      &(alpha),
      (A_vals.ptr(aDomain.lo()) + (A_vals.accessor.strides[0] / sizeof(double)) * loopIdx),
      (A_vals.accessor.strides[1] / sizeof(double))
    ));
  }
}

void computeLegion(Context ctx, Runtime* runtime, LogicalRegion A, LogicalRegion B, LogicalRegion C, LogicalPartition APartition, LogicalPartition BPartition, LogicalPartition CPartition) {

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
  task_4Args taskArgsRaw;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_4Args));
  IndexLauncher launcher = IndexLauncher(taskID(4), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(AReq);
  launcher.add_region_requirement(BReq);
  launcher.add_region_requirement(CReq);
  launcher.tag = launcher.tag | TACOMapper::UNTRACK_VALID_REGIONS;
  runtime->execute_index_space(ctx, launcher);

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
  {
    TaskVariantRegistrar registrar(taskID(3), "task_3");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_3>(registrar, "task_3");
  }
  {
    TaskVariantRegistrar registrar(taskID(4), "task_4");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_4>(registrar, "task_4");
  }
}
