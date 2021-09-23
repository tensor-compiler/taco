#include "cblas.h"
#include "taco_legion_header.h"
#include "taco_mapper.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
using namespace Legion;
typedef FieldAccessor<READ_ONLY,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorROdouble2;
typedef ReductionAccessor<SumReduction<double>,true,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorReducedouble2;

struct task_1Args {
  int32_t sfID;
  int32_t c;
  int32_t rpoc;
};
struct task_2Args {
  int32_t sfID;
  int32_t c;
  int32_t rpoc;
};
struct task_3Args {
  int32_t sfID;
  int32_t c;
  int32_t rpoc;
};
struct task_4Args {
  int32_t in;
  int32_t jn;
  int32_t k1s;
  int32_t rpoc3;
};
struct task_5Args {
  int32_t B1_dimension;
  int32_t B2_dimension;
  int32_t C2_dimension;
  int32_t c;
  int32_t rpoc;
  int32_t rpoc3;
};

std::vector<LogicalPartition> partitionForplaceLegionA(Context ctx, Runtime* runtime, LogicalRegion A, int32_t rpoc) {
  int A1_dimension = runtime->get_index_space_domain(get_index_space(A)).hi()[0] + 1;
  int A2_dimension = runtime->get_index_space_domain(get_index_space(A)).hi()[1] + 1;
  auto A_index_space = get_index_space(A);

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((rpoc - 1), (rpoc - 1), 0);
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  auto ADomain = runtime->get_index_space_domain(ctx, A_index_space);
  DomainPointColoring AColoring = DomainPointColoring();
  for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    Point<2> AStart = Point<2>((in * ((A1_dimension + (rpoc - 1)) / rpoc) + 0 / rpoc), (jn * ((A2_dimension + (rpoc - 1)) / rpoc) + 0 / rpoc));
    Point<2> AEnd = Point<2>(TACO_MIN((in * ((A1_dimension + (rpoc - 1)) / rpoc) + ((A1_dimension + (rpoc - 1)) / rpoc - 1)), ADomain.hi()[0]), TACO_MIN((jn * ((A2_dimension + (rpoc - 1)) / rpoc) + ((A2_dimension + (rpoc - 1)) / rpoc - 1)), ADomain.hi()[1]));
    Rect<2> ARect = Rect<2>(AStart, AEnd);
    if (!ADomain.contains(ARect.lo) || !ADomain.contains(ARect.hi)) {
      ARect = ARect.make_empty();
    }
    AColoring[(*itr)] = ARect;
  }
  auto APartition = runtime->create_index_partition(ctx, A_index_space, domain, AColoring, LEGION_COMPUTE_KIND);
  std::vector<LogicalPartition> computePartitions = std::vector<LogicalPartition>();
  computePartitions.push_back(runtime->get_logical_partition(ctx, get_logical_region(A), APartition));
  return computePartitions;
}

void task_1(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion A = regions[0];

  int32_t distFused = task->index_point[0];
  task_1Args* args = (task_1Args*)(task->args);
  int32_t c = args->c;
  int32_t rpoc = args->rpoc;


  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
  int32_t kn = getIndexPoint(task, 2);
}

void placeLegionA(Context ctx, Runtime* runtime, LogicalRegion A, LogicalPartition APartition, int32_t rpoc, int32_t c) {

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((rpoc - 1), (rpoc - 1), 0);
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  RegionRequirement AReq = RegionRequirement(APartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(A));
  AReq.add_field(FID_VAL);
  std::vector<int> dims = std::vector<int>();
  dims.push_back(rpoc);
  dims.push_back(rpoc);
  dims.push_back(c);
  registerPlacementShardingFunctor(ctx, runtime, shardingID(1), dims);
  task_1Args taskArgsRaw;
  taskArgsRaw.sfID = shardingID(1);
  taskArgsRaw.c = c;
  taskArgsRaw.rpoc = rpoc;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_1Args));
  IndexLauncher launcher = IndexLauncher(taskID(1), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(AReq);
  launcher.tag = TACOMapper::PLACEMENT_SHARD;
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();

}

std::vector<LogicalPartition> partitionForplaceLegionB(Context ctx, Runtime* runtime, LogicalRegion B, int32_t rpoc) {
  int B1_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[0] + 1;
  int B2_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[1] + 1;
  auto B_index_space = get_index_space(B);

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((rpoc - 1), (rpoc - 1), 0);
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  auto BDomain = runtime->get_index_space_domain(ctx, B_index_space);
  DomainPointColoring BColoring = DomainPointColoring();
  for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    Point<2> BStart = Point<2>((in * ((B1_dimension + (rpoc - 1)) / rpoc) + 0 / rpoc), (jn * ((B2_dimension + (rpoc - 1)) / rpoc) + 0 / rpoc));
    Point<2> BEnd = Point<2>(TACO_MIN((in * ((B1_dimension + (rpoc - 1)) / rpoc) + ((B1_dimension + (rpoc - 1)) / rpoc - 1)), BDomain.hi()[0]), TACO_MIN((jn * ((B2_dimension + (rpoc - 1)) / rpoc) + ((B2_dimension + (rpoc - 1)) / rpoc - 1)), BDomain.hi()[1]));
    Rect<2> BRect = Rect<2>(BStart, BEnd);
    if (!BDomain.contains(BRect.lo) || !BDomain.contains(BRect.hi)) {
      BRect = BRect.make_empty();
    }
    BColoring[(*itr)] = BRect;
  }
  auto BPartition = runtime->create_index_partition(ctx, B_index_space, domain, BColoring, LEGION_COMPUTE_KIND);
  std::vector<LogicalPartition> computePartitions = std::vector<LogicalPartition>();
  computePartitions.push_back(runtime->get_logical_partition(ctx, get_logical_region(B), BPartition));
  return computePartitions;
}

void task_2(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion B = regions[0];

  int32_t distFused = task->index_point[0];
  task_2Args* args = (task_2Args*)(task->args);
  int32_t c = args->c;
  int32_t rpoc = args->rpoc;


  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
  int32_t kn = getIndexPoint(task, 2);
}

void placeLegionB(Context ctx, Runtime* runtime, LogicalRegion B, LogicalPartition BPartition, int32_t rpoc, int32_t c) {

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((rpoc - 1), (rpoc - 1), 0);
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  RegionRequirement BReq = RegionRequirement(BPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(B));
  BReq.add_field(FID_VAL);
  std::vector<int> dims = std::vector<int>();
  dims.push_back(rpoc);
  dims.push_back(rpoc);
  dims.push_back(c);
  registerPlacementShardingFunctor(ctx, runtime, shardingID(3), dims);
  task_2Args taskArgsRaw;
  taskArgsRaw.sfID = shardingID(3);
  taskArgsRaw.c = c;
  taskArgsRaw.rpoc = rpoc;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_2Args));
  IndexLauncher launcher = IndexLauncher(taskID(2), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(BReq);
  launcher.tag = TACOMapper::PLACEMENT_SHARD;
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();

}

std::vector<LogicalPartition> partitionForplaceLegionC(Context ctx, Runtime* runtime, LogicalRegion C, int32_t rpoc) {
  int C1_dimension = runtime->get_index_space_domain(get_index_space(C)).hi()[0] + 1;
  int C2_dimension = runtime->get_index_space_domain(get_index_space(C)).hi()[1] + 1;
  auto C_index_space = get_index_space(C);

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((rpoc - 1), (rpoc - 1), 0);
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  auto CDomain = runtime->get_index_space_domain(ctx, C_index_space);
  DomainPointColoring CColoring = DomainPointColoring();
  for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    Point<2> CStart = Point<2>((in * ((C1_dimension + (rpoc - 1)) / rpoc) + 0 / rpoc), (jn * ((C2_dimension + (rpoc - 1)) / rpoc) + 0 / rpoc));
    Point<2> CEnd = Point<2>(TACO_MIN((in * ((C1_dimension + (rpoc - 1)) / rpoc) + ((C1_dimension + (rpoc - 1)) / rpoc - 1)), CDomain.hi()[0]), TACO_MIN((jn * ((C2_dimension + (rpoc - 1)) / rpoc) + ((C2_dimension + (rpoc - 1)) / rpoc - 1)), CDomain.hi()[1]));
    Rect<2> CRect = Rect<2>(CStart, CEnd);
    if (!CDomain.contains(CRect.lo) || !CDomain.contains(CRect.hi)) {
      CRect = CRect.make_empty();
    }
    CColoring[(*itr)] = CRect;
  }
  auto CPartition = runtime->create_index_partition(ctx, C_index_space, domain, CColoring, LEGION_COMPUTE_KIND);
  std::vector<LogicalPartition> computePartitions = std::vector<LogicalPartition>();
  computePartitions.push_back(runtime->get_logical_partition(ctx, get_logical_region(C), CPartition));
  return computePartitions;
}

void task_3(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion C = regions[0];

  int32_t distFused = task->index_point[0];
  task_3Args* args = (task_3Args*)(task->args);
  int32_t c = args->c;
  int32_t rpoc = args->rpoc;


  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
  int32_t kn = getIndexPoint(task, 2);
}

void placeLegionC(Context ctx, Runtime* runtime, LogicalRegion C, LogicalPartition CPartition, int32_t rpoc, int32_t c) {

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((rpoc - 1), (rpoc - 1), 0);
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  RegionRequirement CReq = RegionRequirement(CPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(C));
  CReq.add_field(FID_VAL);
  std::vector<int> dims = std::vector<int>();
  dims.push_back(rpoc);
  dims.push_back(rpoc);
  dims.push_back(c);
  registerPlacementShardingFunctor(ctx, runtime, shardingID(5), dims);
  task_3Args taskArgsRaw;
  taskArgsRaw.sfID = shardingID(5);
  taskArgsRaw.c = c;
  taskArgsRaw.rpoc = rpoc;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_3Args));
  IndexLauncher launcher = IndexLauncher(taskID(3), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(CReq);
  launcher.tag = TACOMapper::PLACEMENT_SHARD;
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();

}

std::vector<LogicalPartition> partitionForcomputeLegion(Context ctx, Runtime* runtime, LogicalRegion A, LogicalRegion B, LogicalRegion C, int32_t rpoc, int32_t c) {
  auto A_index_space = get_index_space(A);
  int B1_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[0] + 1;
  int B2_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[1] + 1;
  auto B_index_space = get_index_space(B);
  int C2_dimension = runtime->get_index_space_domain(get_index_space(C)).hi()[1] + 1;
  auto C_index_space = get_index_space(C);

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((rpoc - 1), (rpoc - 1), (c - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  auto ADomain = runtime->get_index_space_domain(ctx, A_index_space);
  auto BDomain = runtime->get_index_space_domain(ctx, B_index_space);
  auto CDomain = runtime->get_index_space_domain(ctx, C_index_space);
  DomainPointColoring AColoring = DomainPointColoring();
  DomainPointColoring BColoring = DomainPointColoring();
  DomainPointColoring CColoring = DomainPointColoring();
  for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    int32_t kn = (*itr)[2];
    Point<2> AStart = Point<2>((in * ((B1_dimension + (rpoc - 1)) / rpoc) + 0 / rpoc), (jn * ((C2_dimension + (rpoc - 1)) / rpoc) + 0 / rpoc));
    Point<2> AEnd = Point<2>(TACO_MIN((in * ((B1_dimension + (rpoc - 1)) / rpoc) + ((B1_dimension + (rpoc - 1)) / rpoc - 1)), ADomain.hi()[0]), TACO_MIN((jn * ((C2_dimension + (rpoc - 1)) / rpoc) + ((C2_dimension + (rpoc - 1)) / rpoc - 1)), ADomain.hi()[1]));
    Rect<2> ARect = Rect<2>(AStart, AEnd);
    if (!ADomain.contains(ARect.lo) || !ADomain.contains(ARect.hi)) {
      ARect = ARect.make_empty();
    }
    AColoring[(*itr)] = ARect;
    Point<2> BStart = Point<2>((in * ((B1_dimension + (rpoc - 1)) / rpoc) + 0 / rpoc), (kn * ((B2_dimension + (c - 1)) / c) + 0 / c));
    Point<2> BEnd = Point<2>(TACO_MIN((in * ((B1_dimension + (rpoc - 1)) / rpoc) + ((B1_dimension + (rpoc - 1)) / rpoc - 1)), BDomain.hi()[0]), TACO_MIN((kn * ((B2_dimension + (c - 1)) / c) + ((B2_dimension + (c - 1)) / c - 1)), BDomain.hi()[1]));
    Rect<2> BRect = Rect<2>(BStart, BEnd);
    if (!BDomain.contains(BRect.lo) || !BDomain.contains(BRect.hi)) {
      BRect = BRect.make_empty();
    }
    BColoring[(*itr)] = BRect;
    Point<2> CStart = Point<2>((kn * ((B2_dimension + (c - 1)) / c) + 0 / c), (jn * ((C2_dimension + (rpoc - 1)) / rpoc) + 0 / rpoc));
    Point<2> CEnd = Point<2>(TACO_MIN((kn * ((B2_dimension + (c - 1)) / c) + ((B2_dimension + (c - 1)) / c - 1)), CDomain.hi()[0]), TACO_MIN((jn * ((C2_dimension + (rpoc - 1)) / rpoc) + ((C2_dimension + (rpoc - 1)) / rpoc - 1)), CDomain.hi()[1]));
    Rect<2> CRect = Rect<2>(CStart, CEnd);
    if (!CDomain.contains(CRect.lo) || !CDomain.contains(CRect.hi)) {
      CRect = CRect.make_empty();
    }
    CColoring[(*itr)] = CRect;
  }
  auto APartition = runtime->create_index_partition(ctx, A_index_space, domain, AColoring, LEGION_ALIASED_COMPLETE_KIND);
  auto BPartition = runtime->create_index_partition(ctx, B_index_space, domain, BColoring, LEGION_ALIASED_COMPLETE_KIND);
  auto CPartition = runtime->create_index_partition(ctx, C_index_space, domain, CColoring, LEGION_ALIASED_COMPLETE_KIND);
  std::vector<LogicalPartition> computePartitions = std::vector<LogicalPartition>();
  computePartitions.push_back(runtime->get_logical_partition(ctx, get_logical_region(A), APartition));
  computePartitions.push_back(runtime->get_logical_partition(ctx, get_logical_region(B), BPartition));
  computePartitions.push_back(runtime->get_logical_partition(ctx, get_logical_region(C), CPartition));
  return computePartitions;
}

void task_4(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion A = regions[0];
  PhysicalRegion B = regions[1];
  PhysicalRegion C = regions[2];

  task_4Args* args = (task_4Args*)(task->args);
  int32_t in = args->in;
  int32_t jn = args->jn;
  int32_t k1s = args->k1s;
  int32_t rpoc3 = args->rpoc3;

  auto A_index_space = get_index_space(A);
  auto B_index_space = get_index_space(B);
  auto C_index_space = get_index_space(C);
  AccessorROdouble2 B_vals(B, FID_VAL);
  AccessorROdouble2 C_vals(C, FID_VAL);
  AccessorReducedouble2 A_vals(A, FID_VAL, LEGION_REDOP_SUM_FLOAT64);

  auto aDomain = runtime->get_index_space_domain(ctx, A_index_space);
  auto bDomain = runtime->get_index_space_domain(ctx, B_index_space);
  auto cDomain = runtime->get_index_space_domain(ctx, C_index_space);
  if (bDomain.get_volume() == 0 || cDomain.get_volume() == 0)
    return ;

  cblas_dgemm(
    CblasRowMajor,
    CblasNoTrans,
    CblasNoTrans,
    (1 + (bDomain.hi()[0] - bDomain.lo()[0])),
    (1 + (cDomain.hi()[1] - cDomain.lo()[1])),
    (1 + (cDomain.hi()[0] - cDomain.lo()[0])),
    1.00000000,
    B_vals.ptr(bDomain.lo()),
    (B_vals.accessor.strides[0] / sizeof(double)),
    C_vals.ptr(cDomain.lo()),
    (C_vals.accessor.strides[0] / sizeof(double)),
    1.00000000,
    A_vals.ptr(aDomain.lo()),
    (A_vals.accessor.strides[0] / sizeof(double))
  );
}

void task_5(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion A = regions[0];
  PhysicalRegion B = regions[1];
  PhysicalRegion C = regions[2];

  int32_t distFused = task->index_point[0];
  task_5Args* args = (task_5Args*)(task->args);
  int32_t B1_dimension = args->B1_dimension;
  int32_t B2_dimension = args->B2_dimension;
  int32_t C2_dimension = args->C2_dimension;
  int32_t c = args->c;
  int32_t rpoc = args->rpoc;
  int32_t rpoc3 = args->rpoc3;

  auto B_index_space = get_index_space(B);
  auto C_index_space = get_index_space(C);

  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
  int32_t kn = getIndexPoint(task, 2);
  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((rpoc3 - 1));
  auto k1sIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(k1sIndexSpace));
  auto BDomain = runtime->get_index_space_domain(ctx, B_index_space);
  auto CDomain = runtime->get_index_space_domain(ctx, C_index_space);
  DomainPointColoring BColoring = DomainPointColoring();
  DomainPointColoring CColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t k1s = (*itr)[0];
    Point<2> BStart = Point<2>((in * ((B1_dimension + (rpoc - 1)) / rpoc) + 0 / rpoc), (kn * ((B2_dimension + (c - 1)) / c) + (((jn + (in + k1s)) % rpoc3) * ((((B2_dimension + (c - 1)) / c - 0 / c) + (rpoc3 - 1)) / rpoc3) + (0 / c) / rpoc3)));
    Point<2> BEnd = Point<2>(TACO_MIN((in * ((B1_dimension + (rpoc - 1)) / rpoc) + ((B1_dimension + (rpoc - 1)) / rpoc - 1)), BDomain.hi()[0]), TACO_MIN((kn * ((B2_dimension + (c - 1)) / c) + (((jn + (in + k1s)) % rpoc3) * ((((B2_dimension + (c - 1)) / c - 0 / c) + (rpoc3 - 1)) / rpoc3) + (((B2_dimension + (c - 1)) / c + (rpoc3 - 1)) / rpoc3 - 1))), BDomain.hi()[1]));
    Rect<2> BRect = Rect<2>(BStart, BEnd);
    if (!BDomain.contains(BRect.lo) || !BDomain.contains(BRect.hi)) {
      BRect = BRect.make_empty();
    }
    BColoring[(*itr)] = BRect;
    Point<2> CStart = Point<2>((kn * ((B2_dimension + (c - 1)) / c) + (((jn + (in + k1s)) % rpoc3) * ((((B2_dimension + (c - 1)) / c - 0 / c) + (rpoc3 - 1)) / rpoc3) + (0 / c) / rpoc3)), (jn * ((C2_dimension + (rpoc - 1)) / rpoc) + 0 / rpoc));
    Point<2> CEnd = Point<2>(TACO_MIN((kn * ((B2_dimension + (c - 1)) / c) + (((jn + (in + k1s)) % rpoc3) * ((((B2_dimension + (c - 1)) / c - 0 / c) + (rpoc3 - 1)) / rpoc3) + (((B2_dimension + (c - 1)) / c + (rpoc3 - 1)) / rpoc3 - 1))), CDomain.hi()[0]), TACO_MIN((jn * ((C2_dimension + (rpoc - 1)) / rpoc) + ((C2_dimension + (rpoc - 1)) / rpoc - 1)), CDomain.hi()[1]));
    Rect<2> CRect = Rect<2>(CStart, CEnd);
    if (!CDomain.contains(CRect.lo) || !CDomain.contains(CRect.hi)) {
      CRect = CRect.make_empty();
    }
    CColoring[(*itr)] = CRect;
  }
  auto BPartition = runtime->create_index_partition(ctx, B_index_space, domain, BColoring, LEGION_DISJOINT_COMPLETE_KIND);
  auto CPartition = runtime->create_index_partition(ctx, C_index_space, domain, CColoring, LEGION_DISJOINT_COMPLETE_KIND);
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t k1s = (*itr);
    RegionRequirement AReq = RegionRequirement(get_logical_region(A), LEGION_REDOP_SUM_FLOAT64, LEGION_SIMULTANEOUS, get_logical_region(A));
    AReq.add_field(FID_VAL);
    auto BsubReg = runtime->get_logical_subregion_by_color(ctx, runtime->get_logical_partition(ctx, get_logical_region(B), BPartition), k1s);
    RegionRequirement BReq = RegionRequirement(BsubReg, READ_ONLY, EXCLUSIVE, get_logical_region(B));
    BReq.add_field(FID_VAL);
    auto CsubReg = runtime->get_logical_subregion_by_color(ctx, runtime->get_logical_partition(ctx, get_logical_region(C), CPartition), k1s);
    RegionRequirement CReq = RegionRequirement(CsubReg, READ_ONLY, EXCLUSIVE, get_logical_region(C));
    CReq.add_field(FID_VAL);
    task_4Args taskArgsRaw;
    taskArgsRaw.in = in;
    taskArgsRaw.jn = jn;
    taskArgsRaw.k1s = k1s;
    taskArgsRaw.rpoc3 = rpoc3;
    TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_4Args));
    TaskLauncher launcher = TaskLauncher(taskID(4), taskArgs);
    launcher.add_region_requirement(AReq);
    launcher.add_region_requirement(BReq);
    launcher.add_region_requirement(CReq);
    launcher.tag |= TACOMapper::UNTRACK_VALID_REGIONS;
    launcher.tag |= TACOMapper::BACKPRESSURE_TASK;
    runtime->execute_task(ctx, launcher);
  }

}

void computeLegion(Context ctx, Runtime* runtime, LogicalRegion A, LogicalRegion B, LogicalRegion C, LogicalPartition APartition, LogicalPartition BPartition, LogicalPartition CPartition, int32_t rpoc, int32_t c, int32_t rpoc3) {
  int B1_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[0] + 1;
  int B2_dimension = runtime->get_index_space_domain(get_index_space(B)).hi()[1] + 1;
  int C2_dimension = runtime->get_index_space_domain(get_index_space(C)).hi()[1] + 1;

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((rpoc - 1), (rpoc - 1), (c - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  RegionRequirement AReq = RegionRequirement(APartition, 0, LEGION_REDOP_SUM_FLOAT64, LEGION_SIMULTANEOUS, get_logical_region(A));
  AReq.add_field(FID_VAL);
  AReq.tag = Mapping::DefaultMapper::VIRTUAL_MAP;
  RegionRequirement BReq = RegionRequirement(BPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(B));
  BReq.add_field(FID_VAL);
  BReq.tag = Mapping::DefaultMapper::VIRTUAL_MAP;
  RegionRequirement CReq = RegionRequirement(CPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(C));
  CReq.add_field(FID_VAL);
  CReq.tag = Mapping::DefaultMapper::VIRTUAL_MAP;
  task_5Args taskArgsRaw;
  taskArgsRaw.B1_dimension = B1_dimension;
  taskArgsRaw.B2_dimension = B2_dimension;
  taskArgsRaw.C2_dimension = C2_dimension;
  taskArgsRaw.c = c;
  taskArgsRaw.rpoc = rpoc;
  taskArgsRaw.rpoc3 = rpoc3;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_5Args));
  IndexLauncher launcher = IndexLauncher(taskID(5), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(AReq);
  launcher.add_region_requirement(BReq);
  launcher.add_region_requirement(CReq);
  runtime->execute_index_space(ctx, launcher);

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
  {
    TaskVariantRegistrar registrar(taskID(3), "task_3");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_3>(registrar, "task_3");
  }
  {
    TaskVariantRegistrar registrar(taskID(4), "task_4");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_4>(registrar, "task_4");
  }
  {
    TaskVariantRegistrar registrar(taskID(5), "task_5");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_inner();
    Runtime::preregister_task_variant<task_5>(registrar, "task_5");
  }
}
