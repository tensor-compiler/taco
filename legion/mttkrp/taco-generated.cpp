#include "taco_legion_header.h"
#include "taco_mapper.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
using namespace Legion;
typedef FieldAccessor<READ_ONLY,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorROdouble2;
typedef ReductionAccessor<SumReduction<double>,true,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorReducedouble2;
typedef FieldAccessor<READ_ONLY,double,3,coord_t,Realm::AffineAccessor<double,3,coord_t>> AccessorROdouble3;

struct task_1Args {
  int32_t gridX;
  int32_t gridY;
  int32_t gridZ;
};
struct task_2Args {
  int32_t A1_dimension;
  int32_t A2_dimension;
  int32_t B1_dimension;
  int32_t B2_dimension;
  int32_t B3_dimension;
  int32_t C1_dimension;
  int32_t C2_dimension;
  int32_t D1_dimension;
  int32_t D2_dimension;
};

void task_1(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion B = regions[0];

  int32_t distFused = task->index_point[0];
  task_1Args* args = (task_1Args*)(task->args);
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;
  int32_t gridZ = args->gridZ;


  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
  int32_t kn = getIndexPoint(task, 2);
}

LogicalPartition placeLegionB(Context ctx, Runtime* runtime, LogicalRegion B, int32_t gridX, int32_t gridY, int32_t gridZ) {
  int B1_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[0] + 1;
  int B2_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[1] + 1;
  int B3_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[2] + 1;
  auto B_index_space = get_index_space(B);

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((gridX - 1), (gridY - 1), (gridZ - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  auto BDomain = runtime->get_index_space_domain(ctx, B_index_space);
  DomainPointColoring BColoring = DomainPointColoring();
  for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    int32_t kn = (*itr)[2];
    Point<3> BStart = Point<3>((in * ((B1_dimension + (gridX - 1)) / gridX) + 0 / gridX), (jn * ((B2_dimension + (gridY - 1)) / gridY) + 0 / gridY), (kn * ((B3_dimension + (gridZ - 1)) / gridZ) + 0 / gridZ));
    Point<3> BEnd = Point<3>(TACO_MIN((in * ((B1_dimension + (gridX - 1)) / gridX) + ((B1_dimension + (gridX - 1)) / gridX - 1)), BDomain.hi()[0]), TACO_MIN((jn * ((B2_dimension + (gridY - 1)) / gridY) + ((B2_dimension + (gridY - 1)) / gridY - 1)), BDomain.hi()[1]), TACO_MIN((kn * ((B3_dimension + (gridZ - 1)) / gridZ) + ((B3_dimension + (gridZ - 1)) / gridZ - 1)), BDomain.hi()[2]));
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
  task_1Args taskArgsRaw;
  taskArgsRaw.gridX = gridX;
  taskArgsRaw.gridY = gridY;
  taskArgsRaw.gridZ = gridZ;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_1Args));
  IndexLauncher launcher = IndexLauncher(taskID(1), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(BReq);
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  return runtime->get_logical_partition(ctx, get_logical_region(B), BPartition);

}

void task_2(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion A = regions[0];
  PhysicalRegion B = regions[1];
  PhysicalRegion C = regions[2];
  PhysicalRegion D = regions[3];

  int32_t distFused = task->index_point[0];
  task_2Args* args = (task_2Args*)(task->args);
  int32_t A1_dimension = args->A1_dimension;
  int32_t A2_dimension = args->A2_dimension;
  int32_t B1_dimension = args->B1_dimension;
  int32_t B2_dimension = args->B2_dimension;
  int32_t B3_dimension = args->B3_dimension;
  int32_t C1_dimension = args->C1_dimension;
  int32_t C2_dimension = args->C2_dimension;
  int32_t D1_dimension = args->D1_dimension;
  int32_t D2_dimension = args->D2_dimension;

  auto B_index_space = get_index_space(B);
  AccessorROdouble2 C_vals(C, FID_VAL);
  AccessorROdouble2 D_vals(D, FID_VAL);
  AccessorROdouble3 B_vals(B, FID_VAL);
  AccessorReducedouble2 A_vals(A, FID_VAL, LEGION_REDOP_SUM_FLOAT64);

  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
  int32_t kn = getIndexPoint(task, 2);
  auto BPartitionBounds = runtime->get_index_space_domain(ctx, B_index_space);
  int64_t BPartitionBounds0lo = BPartitionBounds.lo()[0];
  int64_t BPartitionBounds0hi = BPartitionBounds.hi()[0];
  int64_t BPartitionBounds1lo = BPartitionBounds.lo()[1];
  int64_t BPartitionBounds1hi = BPartitionBounds.hi()[1];
  int64_t BPartitionBounds2lo = BPartitionBounds.lo()[2];
  int64_t BPartitionBounds2hi = BPartitionBounds.hi()[2];
  #pragma omp parallel for schedule(static)
  for (int32_t ii = 0; ii < ((((BPartitionBounds0hi - BPartitionBounds0lo) + 1) + 3) / 4); ii++) {
    #pragma clang loop interleave(enable) vectorize(enable)
    for (int32_t io = 0; io < 4; io++) {
      int32_t il = ii * 4 + io;
      int32_t i = il + BPartitionBounds0lo;
      if (i >= B1_dimension)
        continue;

      if (i >= (in + 1) * ((BPartitionBounds0hi - BPartitionBounds0lo) + 1))
        continue;

      for (int32_t jl = 0; jl < ((BPartitionBounds1hi - BPartitionBounds1lo) + 1); jl++) {
        int32_t j = jl + BPartitionBounds1lo;
        int32_t jB = i * B2_dimension + j;
        if (j >= C1_dimension)
          continue;

        if (j >= (jn + 1) * ((BPartitionBounds1hi - BPartitionBounds1lo) + 1))
          continue;

        for (int32_t kl = 0; kl < ((BPartitionBounds2hi - BPartitionBounds2lo) + 1); kl++) {
          int32_t k = kl + BPartitionBounds2lo;
          Point<3> B_access_point = Point<3>(i, j, k);
          if (k >= D1_dimension)
            continue;

          if (k >= (kn + 1) * ((BPartitionBounds2hi - BPartitionBounds2lo) + 1))
            continue;

          for (int32_t l = 0; l < D2_dimension; l++) {
            Point<2> A_access_point = Point<2>(i, l);
            Point<2> C_access_point = Point<2>(j, l);
            Point<2> D_access_point = Point<2>(k, l);
            A_vals[A_access_point] <<= (B_vals[B_access_point] * C_vals[C_access_point]) * D_vals[D_access_point];
          }
        }
      }
    }
  }
}

void computeLegion(Context ctx, Runtime* runtime, LogicalRegion A, LogicalRegion B, LogicalRegion C, LogicalRegion D, LogicalPartition BPartition) {
  int A1_dimension = runtime->get_index_space_domain(get_index_space(A)).hi()[0] + 1;
  int A2_dimension = runtime->get_index_space_domain(get_index_space(A)).hi()[1] + 1;
  auto A_index_space = get_index_space(A);
  int B1_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[0] + 1;
  int B2_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[1] + 1;
  int B3_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[2] + 1;
  int C1_dimension = runtime->get_index_space_domain(get_index_space(C)).hi()[0] + 1;
  int C2_dimension = runtime->get_index_space_domain(get_index_space(C)).hi()[1] + 1;
  auto C_index_space = get_index_space(C);
  int D1_dimension = runtime->get_index_space_domain(get_index_space(D)).hi()[0] + 1;
  int D2_dimension = runtime->get_index_space_domain(get_index_space(D)).hi()[1] + 1;
  auto D_index_space = get_index_space(D);

  DomainT<3> domain = runtime->get_index_partition_color_space(ctx, get_index_partition(BPartition));
  auto ADomain = runtime->get_index_space_domain(ctx, A_index_space);
  auto CDomain = runtime->get_index_space_domain(ctx, C_index_space);
  auto DDomain = runtime->get_index_space_domain(ctx, D_index_space);
  DomainPointColoring AColoring = DomainPointColoring();
  DomainPointColoring CColoring = DomainPointColoring();
  DomainPointColoring DColoring = DomainPointColoring();
  for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
    DomainPoint domPoint = (*itr);
    auto BPartitionBounds = runtime->get_index_space_domain(runtime->get_logical_subregion_by_color(ctx, BPartition, domPoint).get_index_space());
    int64_t BPartitionBounds0lo = BPartitionBounds.lo()[0];
    int64_t BPartitionBounds0hi = BPartitionBounds.hi()[0];
    int64_t BPartitionBounds1lo = BPartitionBounds.lo()[1];
    int64_t BPartitionBounds1hi = BPartitionBounds.hi()[1];
    int64_t BPartitionBounds2lo = BPartitionBounds.lo()[2];
    int64_t BPartitionBounds2hi = BPartitionBounds.hi()[2];
    Point<2> AStart = Point<2>(BPartitionBounds0lo, 0);
    Point<2> AEnd = Point<2>(TACO_MIN(((((BPartitionBounds0hi - BPartitionBounds0lo) + 1) - 1) + BPartitionBounds0lo), ADomain.hi()[0]), TACO_MIN(D2_dimension, ADomain.hi()[1]));
    Rect<2> ARect = Rect<2>(AStart, AEnd);
    if (!ADomain.contains(ARect.lo) || !ADomain.contains(ARect.hi)) {
      ARect = ARect.make_empty();
    }
    AColoring[(*itr)] = ARect;
    Point<2> CStart = Point<2>(BPartitionBounds1lo, 0);
    Point<2> CEnd = Point<2>(TACO_MIN(((((BPartitionBounds1hi - BPartitionBounds1lo) + 1) - 1) + BPartitionBounds1lo), CDomain.hi()[0]), TACO_MIN(D2_dimension, CDomain.hi()[1]));
    Rect<2> CRect = Rect<2>(CStart, CEnd);
    if (!CDomain.contains(CRect.lo) || !CDomain.contains(CRect.hi)) {
      CRect = CRect.make_empty();
    }
    CColoring[(*itr)] = CRect;
    Point<2> DStart = Point<2>(BPartitionBounds2lo, 0);
    Point<2> DEnd = Point<2>(TACO_MIN(((((BPartitionBounds2hi - BPartitionBounds2lo) + 1) - 1) + BPartitionBounds2lo), DDomain.hi()[0]), TACO_MIN(D2_dimension, DDomain.hi()[1]));
    Rect<2> DRect = Rect<2>(DStart, DEnd);
    if (!DDomain.contains(DRect.lo) || !DDomain.contains(DRect.hi)) {
      DRect = DRect.make_empty();
    }
    DColoring[(*itr)] = DRect;
  }
  auto APartition = runtime->create_index_partition(ctx, A_index_space, domain, AColoring, LEGION_ALIASED_COMPLETE_KIND);
  auto CPartition = runtime->create_index_partition(ctx, C_index_space, domain, CColoring, LEGION_ALIASED_COMPLETE_KIND);
  auto DPartition = runtime->create_index_partition(ctx, D_index_space, domain, DColoring, LEGION_ALIASED_COMPLETE_KIND);
  LogicalPartition ALogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(A), APartition);
  RegionRequirement AReq = RegionRequirement(ALogicalPartition, 0, LEGION_REDOP_SUM_FLOAT64, LEGION_SIMULTANEOUS, get_logical_region(A));
  AReq.add_field(FID_VAL);
  RegionRequirement BReq = RegionRequirement(BPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(B));
  BReq.add_field(FID_VAL);
  LogicalPartition CLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(C), CPartition);
  RegionRequirement CReq = RegionRequirement(CLogicalPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(C));
  CReq.add_field(FID_VAL);
  LogicalPartition DLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(D), DPartition);
  RegionRequirement DReq = RegionRequirement(DLogicalPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(D));
  DReq.add_field(FID_VAL);
  task_2Args taskArgsRaw;
  taskArgsRaw.A1_dimension = A1_dimension;
  taskArgsRaw.A2_dimension = A2_dimension;
  taskArgsRaw.B1_dimension = B1_dimension;
  taskArgsRaw.B2_dimension = B2_dimension;
  taskArgsRaw.B3_dimension = B3_dimension;
  taskArgsRaw.C1_dimension = C1_dimension;
  taskArgsRaw.C2_dimension = C2_dimension;
  taskArgsRaw.D1_dimension = D1_dimension;
  taskArgsRaw.D2_dimension = D2_dimension;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_2Args));
  IndexLauncher launcher = IndexLauncher(taskID(2), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(AReq);
  launcher.add_region_requirement(BReq);
  launcher.add_region_requirement(CReq);
  launcher.add_region_requirement(DReq);
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
