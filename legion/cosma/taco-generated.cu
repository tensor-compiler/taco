#include "cublas_v2.h"
#include "cudalibs.h"
#include "leaf_kernels.cuh"
#include "taco_legion_header.h"
#include "taco_mapper.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
using namespace Legion;
typedef FieldAccessor<READ_ONLY,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorROdouble2;
typedef ReductionAccessor<SumReduction<double>,true,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorReducedouble2;

struct task_1Args {
  int32_t sfID;
  int32_t gx;
  int32_t gy;
  int32_t gz;
};
struct task_2Args {
  int32_t sfID;
  int32_t gx;
  int32_t gy;
  int32_t gz;
};
struct task_3Args {
  int32_t sfID;
  int32_t gx;
  int32_t gy;
  int32_t gz;
};
struct task_4Args {
  int32_t gx;
  int32_t gy;
  int32_t gz;
};

LogicalPartition partitionLegionA(Context ctx, Runtime* runtime, LogicalRegion a, int32_t gx, int32_t gy) {
  int a1_dimension = runtime->get_index_space_domain(get_index_space(a)).hi()[0] + 1;
  int a2_dimension = runtime->get_index_space_domain(get_index_space(a)).hi()[1] + 1;
  auto a_index_space = get_index_space(a);

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gx - 1), (gy - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  auto aDomain = runtime->get_index_space_domain(ctx, a_index_space);
  DomainPointColoring aColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    Point<2> aStart = Point<2>((in * ((a1_dimension + (gx - 1)) / gx) + 0 / gx), (jn * ((a2_dimension + (gy - 1)) / gy) + 0 / gy));
    Point<2> aEnd = Point<2>(TACO_MIN((in * ((a1_dimension + (gx - 1)) / gx) + ((a1_dimension + (gx - 1)) / gx - 1)),aDomain.hi()[0]), TACO_MIN((jn * ((a2_dimension + (gy - 1)) / gy) + ((a2_dimension + (gy - 1)) / gy - 1)),aDomain.hi()[1]));
    Rect<2> aRect = Rect<2>(aStart, aEnd);
    if (!aDomain.contains(aRect.lo) || !aDomain.contains(aRect.hi)) {
      aRect = aRect.make_empty();
    }
    aColoring[(*itr)] = aRect;
  }
  auto aPartition = runtime->create_index_partition(ctx, a_index_space, domain, aColoring, LEGION_DISJOINT_COMPLETE_KIND);
  return runtime->get_logical_partition(ctx, get_logical_region(a), aPartition);
}

LogicalPartition partitionLegionB(Context ctx, Runtime* runtime, LogicalRegion b, int32_t gx, int32_t gz) {
  int b1_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[0] + 1;
  int b2_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[1] + 1;
  auto b_index_space = get_index_space(b);

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gx - 1), (gz - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  auto bDomain = runtime->get_index_space_domain(ctx, b_index_space);
  DomainPointColoring bColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    Point<2> bStart = Point<2>((in * ((b1_dimension + (gx - 1)) / gx) + 0 / gx), (jn * ((b2_dimension + (gz - 1)) / gz) + 0 / gz));
    Point<2> bEnd = Point<2>(TACO_MIN((in * ((b1_dimension + (gx - 1)) / gx) + ((b1_dimension + (gx - 1)) / gx - 1)),bDomain.hi()[0]), TACO_MIN((jn * ((b2_dimension + (gz - 1)) / gz) + ((b2_dimension + (gz - 1)) / gz - 1)),bDomain.hi()[1]));
    Rect<2> bRect = Rect<2>(bStart, bEnd);
    if (!bDomain.contains(bRect.lo) || !bDomain.contains(bRect.hi)) {
      bRect = bRect.make_empty();
    }
    bColoring[(*itr)] = bRect;
  }
  auto bPartition = runtime->create_index_partition(ctx, b_index_space, domain, bColoring, LEGION_DISJOINT_COMPLETE_KIND);
  return runtime->get_logical_partition(ctx, get_logical_region(b), bPartition);
}

LogicalPartition partitionLegionC(Context ctx, Runtime* runtime, LogicalRegion c, int32_t gz, int32_t gy) {
  int c1_dimension = runtime->get_index_space_domain(get_index_space(c)).hi()[0] + 1;
  int c2_dimension = runtime->get_index_space_domain(get_index_space(c)).hi()[1] + 1;
  auto c_index_space = get_index_space(c);

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gz - 1), (gy - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  auto cDomain = runtime->get_index_space_domain(ctx, c_index_space);
  DomainPointColoring cColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    Point<2> cStart = Point<2>((in * ((c1_dimension + (gz - 1)) / gz) + 0 / gz), (jn * ((c2_dimension + (gy - 1)) / gy) + 0 / gy));
    Point<2> cEnd = Point<2>(TACO_MIN((in * ((c1_dimension + (gz - 1)) / gz) + ((c1_dimension + (gz - 1)) / gz - 1)),cDomain.hi()[0]), TACO_MIN((jn * ((c2_dimension + (gy - 1)) / gy) + ((c2_dimension + (gy - 1)) / gy - 1)),cDomain.hi()[1]));
    Rect<2> cRect = Rect<2>(cStart, cEnd);
    if (!cDomain.contains(cRect.lo) || !cDomain.contains(cRect.hi)) {
      cRect = cRect.make_empty();
    }
    cColoring[(*itr)] = cRect;
  }
  auto cPartition = runtime->create_index_partition(ctx, c_index_space, domain, cColoring, LEGION_DISJOINT_COMPLETE_KIND);
  return runtime->get_logical_partition(ctx, get_logical_region(c), cPartition);
}

void task_1(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a = regions[0];

  int32_t distFused = task->index_point[0];
  task_1Args* args = (task_1Args*)(task->args);
  int32_t gx = args->gx;
  int32_t gy = args->gy;
  int32_t gz = args->gz;


  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
  int32_t kn = getIndexPoint(task, 2);
}

LogicalPartition placeLegionA(Context ctx, Runtime* runtime, LogicalRegion a, int32_t gx, int32_t gy, int32_t gz) {
  int a1_dimension = runtime->get_index_space_domain(get_index_space(a)).hi()[0] + 1;
  int a2_dimension = runtime->get_index_space_domain(get_index_space(a)).hi()[1] + 1;
  auto a_index_space = get_index_space(a);

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((gx - 1), (gy - 1), 0);
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  auto aDomain = runtime->get_index_space_domain(ctx, a_index_space);
  DomainPointColoring aColoring = DomainPointColoring();
  for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    Point<2> aStart = Point<2>((in * ((a1_dimension + (gx - 1)) / gx) + 0 / gx), (jn * ((a2_dimension + (gy - 1)) / gy) + 0 / gy));
    Point<2> aEnd = Point<2>(TACO_MIN((in * ((a1_dimension + (gx - 1)) / gx) + ((a1_dimension + (gx - 1)) / gx - 1)),aDomain.hi()[0]), TACO_MIN((jn * ((a2_dimension + (gy - 1)) / gy) + ((a2_dimension + (gy - 1)) / gy - 1)),aDomain.hi()[1]));
    Rect<2> aRect = Rect<2>(aStart, aEnd);
    if (!aDomain.contains(aRect.lo) || !aDomain.contains(aRect.hi)) {
      aRect = aRect.make_empty();
    }
    aColoring[(*itr)] = aRect;
  }
  auto aPartition = runtime->create_index_partition(ctx, a_index_space, domain, aColoring, LEGION_COMPUTE_KIND);
  LogicalPartition aLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(a), aPartition);
  RegionRequirement aReq = RegionRequirement(aLogicalPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(a));
  aReq.add_field(FID_VAL);
  std::vector<int> dims = std::vector<int>();
  dims.push_back(gx);
  dims.push_back(gy);
  dims.push_back(gz);
  registerPlacementShardingFunctor(ctx, runtime, shardingID(3), dims);
  task_1Args taskArgsRaw;
  taskArgsRaw.sfID = shardingID(3);
  taskArgsRaw.gx = gx;
  taskArgsRaw.gy = gy;
  taskArgsRaw.gz = gz;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_1Args));
  IndexLauncher launcher = IndexLauncher(taskID(1), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(aReq);
  launcher.tag = TACOMapper::PLACEMENT_SHARD;
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  return runtime->get_logical_partition(ctx, get_logical_region(a), aPartition);

}

void task_2(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion b = regions[0];

  int32_t distFused = task->index_point[0];
  task_2Args* args = (task_2Args*)(task->args);
  int32_t gx = args->gx;
  int32_t gy = args->gy;
  int32_t gz = args->gz;


  int32_t in = getIndexPoint(task, 0);
  int32_t kn = getIndexPoint(task, 1);
  int32_t jn = getIndexPoint(task, 2);
}

LogicalPartition placeLegionB(Context ctx, Runtime* runtime, LogicalRegion b, int32_t gx, int32_t gz, int32_t gy) {
  int b1_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[0] + 1;
  int b2_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[1] + 1;
  auto b_index_space = get_index_space(b);

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((gx - 1), 0, (gz - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  auto bDomain = runtime->get_index_space_domain(ctx, b_index_space);
  DomainPointColoring bColoring = DomainPointColoring();
  for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[2];
    Point<2> bStart = Point<2>((in * ((b1_dimension + (gx - 1)) / gx) + 0 / gx), (jn * ((b2_dimension + (gz - 1)) / gz) + 0 / gz));
    Point<2> bEnd = Point<2>(TACO_MIN((in * ((b1_dimension + (gx - 1)) / gx) + ((b1_dimension + (gx - 1)) / gx - 1)),bDomain.hi()[0]), TACO_MIN((jn * ((b2_dimension + (gz - 1)) / gz) + ((b2_dimension + (gz - 1)) / gz - 1)),bDomain.hi()[1]));
    Rect<2> bRect = Rect<2>(bStart, bEnd);
    if (!bDomain.contains(bRect.lo) || !bDomain.contains(bRect.hi)) {
      bRect = bRect.make_empty();
    }
    bColoring[(*itr)] = bRect;
  }
  auto bPartition = runtime->create_index_partition(ctx, b_index_space, domain, bColoring, LEGION_COMPUTE_KIND);
  LogicalPartition bLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(b), bPartition);
  RegionRequirement bReq = RegionRequirement(bLogicalPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(b));
  bReq.add_field(FID_VAL);
  std::vector<int> dims = std::vector<int>();
  dims.push_back(gx);
  dims.push_back(gy);
  dims.push_back(gz);
  registerPlacementShardingFunctor(ctx, runtime, shardingID(4), dims);
  task_2Args taskArgsRaw;
  taskArgsRaw.sfID = shardingID(4);
  taskArgsRaw.gx = gx;
  taskArgsRaw.gy = gy;
  taskArgsRaw.gz = gz;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_2Args));
  IndexLauncher launcher = IndexLauncher(taskID(2), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(bReq);
  launcher.tag = TACOMapper::PLACEMENT_SHARD;
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  return runtime->get_logical_partition(ctx, get_logical_region(b), bPartition);

}

void task_3(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion c = regions[0];

  int32_t distFused = task->index_point[0];
  task_3Args* args = (task_3Args*)(task->args);
  int32_t gx = args->gx;
  int32_t gy = args->gy;
  int32_t gz = args->gz;


  int32_t kn = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
  int32_t in = getIndexPoint(task, 2);
}

LogicalPartition placeLegionC(Context ctx, Runtime* runtime, LogicalRegion c, int32_t gy, int32_t gz, int32_t gx) {
  int c1_dimension = runtime->get_index_space_domain(get_index_space(c)).hi()[0] + 1;
  int c2_dimension = runtime->get_index_space_domain(get_index_space(c)).hi()[1] + 1;
  auto c_index_space = get_index_space(c);

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>(0, (gy - 1), (gz - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  auto cDomain = runtime->get_index_space_domain(ctx, c_index_space);
  DomainPointColoring cColoring = DomainPointColoring();
  for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
    int32_t jn = (*itr)[1];
    int32_t in = (*itr)[2];
    Point<2> cStart = Point<2>((in * ((c1_dimension + (gz - 1)) / gz) + 0 / gz), (jn * ((c2_dimension + (gy - 1)) / gy) + 0 / gy));
    Point<2> cEnd = Point<2>(TACO_MIN((in * ((c1_dimension + (gz - 1)) / gz) + ((c1_dimension + (gz - 1)) / gz - 1)),cDomain.hi()[0]), TACO_MIN((jn * ((c2_dimension + (gy - 1)) / gy) + ((c2_dimension + (gy - 1)) / gy - 1)),cDomain.hi()[1]));
    Rect<2> cRect = Rect<2>(cStart, cEnd);
    if (!cDomain.contains(cRect.lo) || !cDomain.contains(cRect.hi)) {
      cRect = cRect.make_empty();
    }
    cColoring[(*itr)] = cRect;
  }
  auto cPartition = runtime->create_index_partition(ctx, c_index_space, domain, cColoring, LEGION_COMPUTE_KIND);
  LogicalPartition cLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(c), cPartition);
  RegionRequirement cReq = RegionRequirement(cLogicalPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(c));
  cReq.add_field(FID_VAL);
  std::vector<int> dims = std::vector<int>();
  dims.push_back(gx);
  dims.push_back(gy);
  dims.push_back(gz);
  registerPlacementShardingFunctor(ctx, runtime, shardingID(5), dims);
  task_3Args taskArgsRaw;
  taskArgsRaw.sfID = shardingID(5);
  taskArgsRaw.gx = gx;
  taskArgsRaw.gy = gy;
  taskArgsRaw.gz = gz;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_3Args));
  IndexLauncher launcher = IndexLauncher(taskID(3), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(cReq);
  launcher.tag = TACOMapper::PLACEMENT_SHARD;
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  return runtime->get_logical_partition(ctx, get_logical_region(c), cPartition);

}

void task_4(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a = regions[0];
  PhysicalRegion b = regions[1];
  PhysicalRegion c = regions[2];

  int32_t distFused = task->index_point[0];
  task_4Args* args = (task_4Args*)(task->args);
  int32_t gx = args->gx;
  int32_t gy = args->gy;
  int32_t gz = args->gz;

  auto a_index_space = get_index_space(a);
  auto b_index_space = get_index_space(b);
  auto c_index_space = get_index_space(c);
  AccessorROdouble2 b_vals(b, FID_VAL);
  AccessorROdouble2 c_vals(c, FID_VAL);
  AccessorReducedouble2 a_vals(a, FID_VAL, LEGION_REDOP_SUM_FLOAT64);

  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
  int32_t kn = getIndexPoint(task, 2);
  auto aDomain = runtime->get_index_space_domain(ctx, a_index_space);
  auto bDomain = runtime->get_index_space_domain(ctx, b_index_space);
  auto cDomain = runtime->get_index_space_domain(ctx, c_index_space);
  if (bDomain.get_volume() == 0 || cDomain.get_volume() == 0)
    return ;

  double alpha = 1.0000000000000000;
  cublasHandle_t handle = getCuBLAS();
  cudaStream_t taskStream = cudaStream_t();
  cudaStreamCreate(&(taskStream));
  CHECK_CUBLAS(cublasSetStream(handle, taskStream));
  CHECK_CUBLAS(cublasDgemm(
    handle,
    CUBLAS_OP_N,
    CUBLAS_OP_N,
    (1 + (cDomain.hi()[1] - cDomain.lo()[1])),
    (1 + (bDomain.hi()[0] - bDomain.lo()[0])),
    (1 + (cDomain.hi()[0] - cDomain.lo()[0])),
    &(alpha),
    c_vals.ptr(cDomain.lo()),
    (c_vals.accessor.strides[0] / sizeof(double)),
    b_vals.ptr(bDomain.lo()),
    (b_vals.accessor.strides[0] / sizeof(double)),
    &(alpha),
    a_vals.ptr(aDomain.lo()),
    (a_vals.accessor.strides[0] / sizeof(double))
  ));
}

void computeLegion(Context ctx, Runtime* runtime, LogicalRegion a, LogicalRegion b, LogicalRegion c, int32_t gx, int32_t gy, int32_t gz) {
  auto a_index_space = get_index_space(a);
  int b1_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[0] + 1;
  int b2_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[1] + 1;
  auto b_index_space = get_index_space(b);
  int c2_dimension = runtime->get_index_space_domain(get_index_space(c)).hi()[1] + 1;
  auto c_index_space = get_index_space(c);

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>((gx - 1), (gy - 1), (gz - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  auto aDomain = runtime->get_index_space_domain(ctx, a_index_space);
  auto bDomain = runtime->get_index_space_domain(ctx, b_index_space);
  auto cDomain = runtime->get_index_space_domain(ctx, c_index_space);
  DomainPointColoring aColoring = DomainPointColoring();
  DomainPointColoring bColoring = DomainPointColoring();
  DomainPointColoring cColoring = DomainPointColoring();
  for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    int32_t kn = (*itr)[2];
    Point<2> aStart = Point<2>((in * ((b1_dimension + (gx - 1)) / gx) + 0 / gx), (jn * ((c2_dimension + (gy - 1)) / gy) + 0 / gy));
    Point<2> aEnd = Point<2>(TACO_MIN((in * ((b1_dimension + (gx - 1)) / gx) + ((b1_dimension + (gx - 1)) / gx - 1)),aDomain.hi()[0]), TACO_MIN((jn * ((c2_dimension + (gy - 1)) / gy) + ((c2_dimension + (gy - 1)) / gy - 1)),aDomain.hi()[1]));
    Rect<2> aRect = Rect<2>(aStart, aEnd);
    if (!aDomain.contains(aRect.lo) || !aDomain.contains(aRect.hi)) {
      aRect = aRect.make_empty();
    }
    aColoring[(*itr)] = aRect;
    Point<2> bStart = Point<2>((in * ((b1_dimension + (gx - 1)) / gx) + 0 / gx), (kn * ((b2_dimension + (gz - 1)) / gz) + 0 / gz));
    Point<2> bEnd = Point<2>(TACO_MIN((in * ((b1_dimension + (gx - 1)) / gx) + ((b1_dimension + (gx - 1)) / gx - 1)),bDomain.hi()[0]), TACO_MIN((kn * ((b2_dimension + (gz - 1)) / gz) + ((b2_dimension + (gz - 1)) / gz - 1)),bDomain.hi()[1]));
    Rect<2> bRect = Rect<2>(bStart, bEnd);
    if (!bDomain.contains(bRect.lo) || !bDomain.contains(bRect.hi)) {
      bRect = bRect.make_empty();
    }
    bColoring[(*itr)] = bRect;
    Point<2> cStart = Point<2>((kn * ((b2_dimension + (gz - 1)) / gz) + 0 / gz), (jn * ((c2_dimension + (gy - 1)) / gy) + 0 / gy));
    Point<2> cEnd = Point<2>(TACO_MIN((kn * ((b2_dimension + (gz - 1)) / gz) + ((b2_dimension + (gz - 1)) / gz - 1)),cDomain.hi()[0]), TACO_MIN((jn * ((c2_dimension + (gy - 1)) / gy) + ((c2_dimension + (gy - 1)) / gy - 1)),cDomain.hi()[1]));
    Rect<2> cRect = Rect<2>(cStart, cEnd);
    if (!cDomain.contains(cRect.lo) || !cDomain.contains(cRect.hi)) {
      cRect = cRect.make_empty();
    }
    cColoring[(*itr)] = cRect;
  }
  auto aPartition = runtime->create_index_partition(ctx, a_index_space, domain, aColoring, LEGION_ALIASED_COMPLETE_KIND);
  auto bPartition = runtime->create_index_partition(ctx, b_index_space, domain, bColoring, LEGION_ALIASED_COMPLETE_KIND);
  auto cPartition = runtime->create_index_partition(ctx, c_index_space, domain, cColoring, LEGION_ALIASED_COMPLETE_KIND);
  LogicalPartition aLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(a), aPartition);
  RegionRequirement aReq = RegionRequirement(aLogicalPartition, 0, LEGION_REDOP_SUM_FLOAT64, LEGION_SIMULTANEOUS, get_logical_region(a));
  aReq.add_field(FID_VAL);
  LogicalPartition bLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(b), bPartition);
  RegionRequirement bReq = RegionRequirement(bLogicalPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(b));
  bReq.add_field(FID_VAL);
  LogicalPartition cLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(c), cPartition);
  RegionRequirement cReq = RegionRequirement(cLogicalPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(c));
  cReq.add_field(FID_VAL);
  task_4Args taskArgsRaw;
  taskArgsRaw.gx = gx;
  taskArgsRaw.gy = gy;
  taskArgsRaw.gz = gz;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_4Args));
  IndexLauncher launcher = IndexLauncher(taskID(4), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(aReq);
  launcher.add_region_requirement(bReq);
  launcher.add_region_requirement(cReq);
  launcher.tag = launcher.tag | TACOMapper::UNTRACK_VALID_REGIONS;
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
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_4>(registrar, "task_4");
  }
}
