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
  int32_t A1_dimension;
  int32_t A2_dimension;
  int32_t A3_dimension;
  int32_t B1_dimension;
  int32_t B2_dimension;
  int32_t B3_dimension;
  int32_t C1_dimension;
  int32_t C2_dimension;
};

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
    Point<3> AEnd = Point<3>(TACO_MIN((in * ((A1_dimension + (pieces - 1)) / pieces) + ((A1_dimension + (pieces - 1)) / pieces - 1)), ADomain.hi()[0]), TACO_MIN(A2_dimension, ADomain.hi()[1]), TACO_MIN(A3_dimension, ADomain.hi()[2]));
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
  PhysicalRegion A = regions[0];
  PhysicalRegion B = regions[1];
  PhysicalRegion C = regions[2];

  int32_t in = task->index_point[0];
  task_2Args* args = (task_2Args*)(task->args);
  int32_t A1_dimension = args->A1_dimension;
  int32_t A2_dimension = args->A2_dimension;
  int32_t A3_dimension = args->A3_dimension;
  int32_t B1_dimension = args->B1_dimension;
  int32_t B2_dimension = args->B2_dimension;
  int32_t B3_dimension = args->B3_dimension;
  int32_t C1_dimension = args->C1_dimension;
  int32_t C2_dimension = args->C2_dimension;

  auto A_index_space = get_index_space(A);
  AccessorROdouble2 C_vals(C, FID_VAL);
  AccessorROdouble3 B_vals(B, FID_VAL);
  AccessorRWdouble3 A_vals(A, FID_VAL);

  auto APartitionBounds = runtime->get_index_space_domain(ctx, A_index_space);
  int64_t APartitionBounds0lo = APartitionBounds.lo()[0];
  int64_t APartitionBounds0hi = APartitionBounds.hi()[0];
  for (int32_t il = 0; il < ((APartitionBounds0hi - APartitionBounds0lo) + 1); il++) {
    int32_t i = il + APartitionBounds0lo;
    if (i >= B1_dimension)
      continue;

    if (i >= (in + 1) * ((APartitionBounds0hi - APartitionBounds0lo) + 1))
      continue;

    for (int32_t j = 0; j < B2_dimension; j++) {
      int32_t jA = i * A2_dimension + j;
      int32_t jB = i * B2_dimension + j;
      for (int32_t l = 0; l < C2_dimension; l++) {
        Point<3> A_access_point = Point<3>(i, j, l);
        for (int32_t k = 0; k < C1_dimension; k++) {
          Point<3> B_access_point = Point<3>(i, j, k);
          Point<2> C_access_point = Point<2>(k, l);
          A_vals[A_access_point] = A_vals[A_access_point] + B_vals[B_access_point] * C_vals[C_access_point];
        }
      }
    }
  }
}

void computeLegion(Context ctx, Runtime* runtime, LogicalRegion A, LogicalRegion B, LogicalRegion C, LogicalPartition APartition) {
  int A1_dimension = runtime->get_index_space_domain(get_index_space(A)).hi()[0] + 1;
  int A2_dimension = runtime->get_index_space_domain(get_index_space(A)).hi()[1] + 1;
  int A3_dimension = runtime->get_index_space_domain(get_index_space(A)).hi()[2] + 1;
  int B1_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[0] + 1;
  int B2_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[1] + 1;
  int B3_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[2] + 1;
  auto B_index_space = get_index_space(B);
  int C1_dimension = runtime->get_index_space_domain(get_index_space(C)).hi()[0] + 1;
  int C2_dimension = runtime->get_index_space_domain(get_index_space(C)).hi()[1] + 1;
  auto C_index_space = get_index_space(C);

  DomainT<1> domain = runtime->get_index_partition_color_space(ctx, get_index_partition(APartition));
  auto BDomain = runtime->get_index_space_domain(ctx, B_index_space);
  auto CDomain = runtime->get_index_space_domain(ctx, C_index_space);
  DomainPointColoring BColoring = DomainPointColoring();
  DomainPointColoring CColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    DomainPoint domPoint = (*itr);
    auto APartitionBounds = runtime->get_index_space_domain(runtime->get_logical_subregion_by_color(ctx, APartition, domPoint).get_index_space());
    int64_t APartitionBounds0lo = APartitionBounds.lo()[0];
    int64_t APartitionBounds0hi = APartitionBounds.hi()[0];
    Point<3> BStart = Point<3>(APartitionBounds0lo, 0, 0);
    Point<3> BEnd = Point<3>(TACO_MIN(((((APartitionBounds0hi - APartitionBounds0lo) + 1) - 1) + APartitionBounds0lo), BDomain.hi()[0]), TACO_MIN(B2_dimension, BDomain.hi()[1]), TACO_MIN(C1_dimension, BDomain.hi()[2]));
    Rect<3> BRect = Rect<3>(BStart, BEnd);
    if (!BDomain.contains(BRect.lo) || !BDomain.contains(BRect.hi)) {
      BRect = BRect.make_empty();
    }
    BColoring[(*itr)] = BRect;
    Point<2> CStart = Point<2>(0, 0);
    Point<2> CEnd = Point<2>(TACO_MIN(C1_dimension, CDomain.hi()[0]), TACO_MIN(C2_dimension, CDomain.hi()[1]));
    Rect<2> CRect = Rect<2>(CStart, CEnd);
    if (!CDomain.contains(CRect.lo) || !CDomain.contains(CRect.hi)) {
      CRect = CRect.make_empty();
    }
    CColoring[(*itr)] = CRect;
  }
  auto BPartition = runtime->create_index_partition(ctx, B_index_space, domain, BColoring, LEGION_ALIASED_COMPLETE_KIND);
  auto CPartition = runtime->create_index_partition(ctx, C_index_space, domain, CColoring, LEGION_ALIASED_COMPLETE_KIND);
  RegionRequirement AReq = RegionRequirement(APartition, 0, READ_WRITE, EXCLUSIVE, get_logical_region(A));
  AReq.add_field(FID_VAL);
  LogicalPartition BLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(B), BPartition);
  RegionRequirement BReq = RegionRequirement(BLogicalPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(B));
  BReq.add_field(FID_VAL);
  LogicalPartition CLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(C), CPartition);
  RegionRequirement CReq = RegionRequirement(CLogicalPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(C));
  CReq.add_field(FID_VAL);
  task_2Args taskArgsRaw;
  taskArgsRaw.A1_dimension = A1_dimension;
  taskArgsRaw.A2_dimension = A2_dimension;
  taskArgsRaw.A3_dimension = A3_dimension;
  taskArgsRaw.B1_dimension = B1_dimension;
  taskArgsRaw.B2_dimension = B2_dimension;
  taskArgsRaw.B3_dimension = B3_dimension;
  taskArgsRaw.C1_dimension = C1_dimension;
  taskArgsRaw.C2_dimension = C2_dimension;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_2Args));
  IndexLauncher launcher = IndexLauncher(taskID(2), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(AReq);
  launcher.add_region_requirement(BReq);
  launcher.add_region_requirement(CReq);
  launcher.tag |= TACOMapper::UNTRACK_VALID_REGIONS;
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();

}
void registerTacoTasks() {
  {
    TaskVariantRegistrar registrar(taskID(1), "task_1");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_1>(registrar, "task_1");
  }
  {
    TaskVariantRegistrar registrar(taskID(2), "task_2");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_2>(registrar, "task_2");
  }
}
