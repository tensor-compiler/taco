#include "cublas_v2.h"
#include "cudalibs.h"
#include "taco_legion_header.h"
#include "taco_mapper.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
using namespace Legion;
typedef FieldAccessor<READ_ONLY,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorROdouble2;
typedef FieldAccessor<READ_WRITE,double,2,coord_t,Realm::AffineAccessor<double,2,coord_t>> AccessorRWdouble2;

struct task_1Args {
};
struct task_2Args {
  int32_t a1_dimension;
  int32_t a2_dimension;
  int32_t gridX;
  int32_t gridY;
};
struct task_3Args {
};
struct task_4Args {
  int32_t b1_dimension;
  int32_t b2_dimension;
  int32_t gridX;
  int32_t gridY;
};
struct task_5Args {
};
struct task_6Args {
  int32_t c1_dimension;
  int32_t c2_dimension;
  int32_t gridX;
  int32_t gridY;
};
struct task_7Args {
  int32_t kio;
};
struct task_8Args {
  int64_t aPartitionBounds0hi;
  int64_t aPartitionBounds0lo;
  int64_t aPartitionBounds1hi;
  int64_t aPartitionBounds1lo;
  int32_t c1_dimension;
  int32_t gridX;
  int32_t ko;
};
struct task_9Args {
  int64_t aPartitionBounds0hi;
  int64_t aPartitionBounds0lo;
  int64_t aPartitionBounds1hi;
  int64_t aPartitionBounds1lo;
  int32_t c1_dimension;
  int32_t gridX;
  int32_t ko;
};
struct task_10Args {
  int32_t c1_dimension;
  int32_t gridX;
};

LogicalPartition partitionLegion(Context ctx, Runtime* runtime, LogicalRegion a, int32_t gridX, int32_t gridY) {
  int a1_dimension = runtime->get_index_space_domain(get_index_space(a)).hi()[0] + 1;
  int a2_dimension = runtime->get_index_space_domain(get_index_space(a)).hi()[1] + 1;
  auto a_index_space = get_index_space(a);

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX * 2 - 1), (gridY * 2 - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  auto aDomain = runtime->get_index_space_domain(ctx, a_index_space);
  DomainPointColoring aColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    Point<2> aStart = Point<2>((in * ((a1_dimension + (gridX * 2 - 1)) / (gridX * 2)) + 0 / (gridX * 2)), (jn * ((a2_dimension + (gridY * 2 - 1)) / (gridY * 2)) + 0 / (gridY * 2)));
    Point<2> aEnd = Point<2>(TACO_MIN((in * ((a1_dimension + (gridX * 2 - 1)) / (gridX * 2)) + ((a1_dimension + (gridX * 2 - 1)) / (gridX * 2) - 1)),aDomain.hi()[0]), TACO_MIN((jn * ((a2_dimension + (gridY * 2 - 1)) / (gridY * 2)) + ((a2_dimension + (gridY * 2 - 1)) / (gridY * 2) - 1)),aDomain.hi()[1]));
    Rect<2> aRect = Rect<2>(aStart, aEnd);
    if (!aDomain.contains(aRect.lo) || !aDomain.contains(aRect.hi)) {
      aRect = aRect.make_empty();
    }
    aColoring[(*itr)] = aRect;
  }
  auto aPartition = runtime->create_index_partition(ctx, a_index_space, domain, aColoring, LEGION_DISJOINT_COMPLETE_KIND);
  return runtime->get_logical_partition(ctx, get_logical_region(a), aPartition);
}

void task_1(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a = regions[0];

  int32_t distFused1 = task->index_point[0];

  int32_t iln = getIndexPoint(task, 0);
  int32_t jln = getIndexPoint(task, 1);
}

void task_2(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a = regions[0];

  int32_t distFused = task->index_point[0];
  task_2Args* args = (task_2Args*)(task->args);
  int32_t a1_dimension = args->a1_dimension;
  int32_t a2_dimension = args->a2_dimension;
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;

  auto a_index_space = get_index_space(a);

  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>(1, 1);
  auto distFused1IndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFused1IndexSpace));
  auto aDomain = runtime->get_index_space_domain(ctx, a_index_space);
  DomainPointColoring aColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int32_t iln = (*itr)[0];
    int32_t jln = (*itr)[1];
    Point<2> aStart = Point<2>((in * ((a1_dimension + (gridX - 1)) / gridX) + (iln * ((((a1_dimension + (gridX - 1)) / gridX - 0 / gridX) + 1) / 2) + (0 / gridX) / 2)), (jn * ((a2_dimension + (gridY - 1)) / gridY) + (jln * ((((a2_dimension + (gridY - 1)) / gridY - 0 / gridY) + 1) / 2) + (0 / gridY) / 2)));
    Point<2> aEnd = Point<2>(TACO_MIN((in * ((a1_dimension + (gridX - 1)) / gridX) + (iln * ((((a1_dimension + (gridX - 1)) / gridX - 0 / gridX) + 1) / 2) + (((a1_dimension + (gridX - 1)) / gridX + 1) / 2 - 1))),aDomain.hi()[0]), TACO_MIN((jn * ((a2_dimension + (gridY - 1)) / gridY) + (jln * ((((a2_dimension + (gridY - 1)) / gridY - 0 / gridY) + 1) / 2) + (((a2_dimension + (gridY - 1)) / gridY + 1) / 2 - 1))),aDomain.hi()[1]));
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
  task_1Args taskArgsRaw;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_1Args));
  IndexLauncher launcher = IndexLauncher(taskID(1), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(aReq);
  launcher.tag = launcher.tag | Mapping::DefaultMapper::SAME_ADDRESS_SPACE;
  runtime->execute_index_space(ctx, launcher);

}

LogicalPartition placeLegionA(Context ctx, Runtime* runtime, LogicalRegion a, int32_t gridX, int32_t gridY) {
  int a1_dimension = runtime->get_index_space_domain(get_index_space(a)).hi()[0] + 1;
  int a2_dimension = runtime->get_index_space_domain(get_index_space(a)).hi()[1] + 1;
  auto a_index_space = get_index_space(a);

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  auto aDomain = runtime->get_index_space_domain(ctx, a_index_space);
  DomainPointColoring aColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    Point<2> aStart = Point<2>((in * ((a1_dimension + (gridX - 1)) / gridX) + 0 / gridX), (jn * ((a2_dimension + (gridY - 1)) / gridY) + 0 / gridY));
    Point<2> aEnd = Point<2>(TACO_MIN((in * ((a1_dimension + (gridX - 1)) / gridX) + ((a1_dimension + (gridX - 1)) / gridX - 1)),aDomain.hi()[0]), TACO_MIN((jn * ((a2_dimension + (gridY - 1)) / gridY) + ((a2_dimension + (gridY - 1)) / gridY - 1)),aDomain.hi()[1]));
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
  aReq.tag = Mapping::DefaultMapper::VIRTUAL_MAP;
  task_2Args taskArgsRaw;
  taskArgsRaw.a1_dimension = a1_dimension;
  taskArgsRaw.a2_dimension = a2_dimension;
  taskArgsRaw.gridX = gridX;
  taskArgsRaw.gridY = gridY;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_2Args));
  IndexLauncher launcher = IndexLauncher(taskID(2), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(aReq);
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  return runtime->get_logical_partition(ctx, get_logical_region(a), aPartition);

}

void task_3(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion b = regions[0];

  int32_t distFused1 = task->index_point[0];

  int32_t iln = getIndexPoint(task, 0);
  int32_t jln = getIndexPoint(task, 1);
}

void task_4(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion b = regions[0];

  int32_t distFused = task->index_point[0];
  task_4Args* args = (task_4Args*)(task->args);
  int32_t b1_dimension = args->b1_dimension;
  int32_t b2_dimension = args->b2_dimension;
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;

  auto b_index_space = get_index_space(b);

  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>(1, 1);
  auto distFused1IndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFused1IndexSpace));
  auto bDomain = runtime->get_index_space_domain(ctx, b_index_space);
  DomainPointColoring bColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int32_t iln = (*itr)[0];
    int32_t jln = (*itr)[1];
    Point<2> bStart = Point<2>((in * ((b1_dimension + (gridX - 1)) / gridX) + (iln * ((((b1_dimension + (gridX - 1)) / gridX - 0 / gridX) + 1) / 2) + (0 / gridX) / 2)), (jn * ((b2_dimension + (gridY - 1)) / gridY) + (jln * ((((b2_dimension + (gridY - 1)) / gridY - 0 / gridY) + 1) / 2) + (0 / gridY) / 2)));
    Point<2> bEnd = Point<2>(TACO_MIN((in * ((b1_dimension + (gridX - 1)) / gridX) + (iln * ((((b1_dimension + (gridX - 1)) / gridX - 0 / gridX) + 1) / 2) + (((b1_dimension + (gridX - 1)) / gridX + 1) / 2 - 1))),bDomain.hi()[0]), TACO_MIN((jn * ((b2_dimension + (gridY - 1)) / gridY) + (jln * ((((b2_dimension + (gridY - 1)) / gridY - 0 / gridY) + 1) / 2) + (((b2_dimension + (gridY - 1)) / gridY + 1) / 2 - 1))),bDomain.hi()[1]));
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
  task_3Args taskArgsRaw;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_3Args));
  IndexLauncher launcher = IndexLauncher(taskID(3), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(bReq);
  launcher.tag = launcher.tag | Mapping::DefaultMapper::SAME_ADDRESS_SPACE;
  runtime->execute_index_space(ctx, launcher);

}

LogicalPartition placeLegionB(Context ctx, Runtime* runtime, LogicalRegion b, int32_t gridX, int32_t gridY) {
  int b1_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[0] + 1;
  int b2_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[1] + 1;
  auto b_index_space = get_index_space(b);

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  auto bDomain = runtime->get_index_space_domain(ctx, b_index_space);
  DomainPointColoring bColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    Point<2> bStart = Point<2>((in * ((b1_dimension + (gridX - 1)) / gridX) + 0 / gridX), (jn * ((b2_dimension + (gridY - 1)) / gridY) + 0 / gridY));
    Point<2> bEnd = Point<2>(TACO_MIN((in * ((b1_dimension + (gridX - 1)) / gridX) + ((b1_dimension + (gridX - 1)) / gridX - 1)),bDomain.hi()[0]), TACO_MIN((jn * ((b2_dimension + (gridY - 1)) / gridY) + ((b2_dimension + (gridY - 1)) / gridY - 1)),bDomain.hi()[1]));
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
  bReq.tag = Mapping::DefaultMapper::VIRTUAL_MAP;
  task_4Args taskArgsRaw;
  taskArgsRaw.b1_dimension = b1_dimension;
  taskArgsRaw.b2_dimension = b2_dimension;
  taskArgsRaw.gridX = gridX;
  taskArgsRaw.gridY = gridY;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_4Args));
  IndexLauncher launcher = IndexLauncher(taskID(4), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(bReq);
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  return runtime->get_logical_partition(ctx, get_logical_region(b), bPartition);

}

void task_5(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion c = regions[0];

  int32_t distFused1 = task->index_point[0];

  int32_t iln = getIndexPoint(task, 0);
  int32_t jln = getIndexPoint(task, 1);
}

void task_6(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion c = regions[0];

  int32_t distFused = task->index_point[0];
  task_6Args* args = (task_6Args*)(task->args);
  int32_t c1_dimension = args->c1_dimension;
  int32_t c2_dimension = args->c2_dimension;
  int32_t gridX = args->gridX;
  int32_t gridY = args->gridY;

  auto c_index_space = get_index_space(c);

  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>(1, 1);
  auto distFused1IndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFused1IndexSpace));
  auto cDomain = runtime->get_index_space_domain(ctx, c_index_space);
  DomainPointColoring cColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int32_t iln = (*itr)[0];
    int32_t jln = (*itr)[1];
    Point<2> cStart = Point<2>((in * ((c1_dimension + (gridX - 1)) / gridX) + (iln * ((((c1_dimension + (gridX - 1)) / gridX - 0 / gridX) + 1) / 2) + (0 / gridX) / 2)), (jn * ((c2_dimension + (gridY - 1)) / gridY) + (jln * ((((c2_dimension + (gridY - 1)) / gridY - 0 / gridY) + 1) / 2) + (0 / gridY) / 2)));
    Point<2> cEnd = Point<2>(TACO_MIN((in * ((c1_dimension + (gridX - 1)) / gridX) + (iln * ((((c1_dimension + (gridX - 1)) / gridX - 0 / gridX) + 1) / 2) + (((c1_dimension + (gridX - 1)) / gridX + 1) / 2 - 1))),cDomain.hi()[0]), TACO_MIN((jn * ((c2_dimension + (gridY - 1)) / gridY) + (jln * ((((c2_dimension + (gridY - 1)) / gridY - 0 / gridY) + 1) / 2) + (((c2_dimension + (gridY - 1)) / gridY + 1) / 2 - 1))),cDomain.hi()[1]));
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
  task_5Args taskArgsRaw;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_5Args));
  IndexLauncher launcher = IndexLauncher(taskID(5), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(cReq);
  launcher.tag = launcher.tag | Mapping::DefaultMapper::SAME_ADDRESS_SPACE;
  runtime->execute_index_space(ctx, launcher);

}

LogicalPartition placeLegionC(Context ctx, Runtime* runtime, LogicalRegion c, int32_t gridX, int32_t gridY) {
  int c1_dimension = runtime->get_index_space_domain(get_index_space(c)).hi()[0] + 1;
  int c2_dimension = runtime->get_index_space_domain(get_index_space(c)).hi()[1] + 1;
  auto c_index_space = get_index_space(c);

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>((gridX - 1), (gridY - 1));
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  auto cDomain = runtime->get_index_space_domain(ctx, c_index_space);
  DomainPointColoring cColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    Point<2> cStart = Point<2>((in * ((c1_dimension + (gridX - 1)) / gridX) + 0 / gridX), (jn * ((c2_dimension + (gridY - 1)) / gridY) + 0 / gridY));
    Point<2> cEnd = Point<2>(TACO_MIN((in * ((c1_dimension + (gridX - 1)) / gridX) + ((c1_dimension + (gridX - 1)) / gridX - 1)),cDomain.hi()[0]), TACO_MIN((jn * ((c2_dimension + (gridY - 1)) / gridY) + ((c2_dimension + (gridY - 1)) / gridY - 1)),cDomain.hi()[1]));
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
  cReq.tag = Mapping::DefaultMapper::VIRTUAL_MAP;
  task_6Args taskArgsRaw;
  taskArgsRaw.c1_dimension = c1_dimension;
  taskArgsRaw.c2_dimension = c2_dimension;
  taskArgsRaw.gridX = gridX;
  taskArgsRaw.gridY = gridY;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_6Args));
  IndexLauncher launcher = IndexLauncher(taskID(6), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(cReq);
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  return runtime->get_logical_partition(ctx, get_logical_region(c), cPartition);

}

void task_7(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a = regions[0];
  PhysicalRegion b = regions[1];
  PhysicalRegion c = regions[2];

  task_7Args* args = (task_7Args*)(task->args);
  int32_t kio = args->kio;

  auto a_index_space = get_index_space(a);
  auto b_index_space = get_index_space(b);
  auto c_index_space = get_index_space(c);
  AccessorROdouble2 b_vals(b, FID_VAL);
  AccessorROdouble2 c_vals(c, FID_VAL);
  AccessorRWdouble2 a_vals(a, FID_VAL);

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

void task_8(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a = regions[0];
  PhysicalRegion b = regions[1];
  PhysicalRegion c = regions[2];

  int32_t distFused1 = task->index_point[0];
  task_8Args* args = (task_8Args*)(task->args);
  int64_t aPartitionBounds0hi = args->aPartitionBounds0hi;
  int64_t aPartitionBounds0lo = args->aPartitionBounds0lo;
  int64_t aPartitionBounds1hi = args->aPartitionBounds1hi;
  int64_t aPartitionBounds1lo = args->aPartitionBounds1lo;
  int32_t c1_dimension = args->c1_dimension;
  int32_t gridX = args->gridX;
  int32_t ko = args->ko;

  auto b_index_space = get_index_space(b);
  auto c_index_space = get_index_space(c);

  int32_t iln = getIndexPoint(task, 0);
  int32_t jln = getIndexPoint(task, 1);
  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>(1);
  auto kioIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(kioIndexSpace));
  auto bDomain = runtime->get_index_space_domain(ctx, b_index_space);
  auto cDomain = runtime->get_index_space_domain(ctx, c_index_space);
  DomainPointColoring bColoring = DomainPointColoring();
  DomainPointColoring cColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t kio = (*itr)[0];
    Point<2> bStart = Point<2>((iln * ((((aPartitionBounds0hi - aPartitionBounds0lo) + 1) + 1) / 2) + aPartitionBounds0lo), (ko * ((c1_dimension + (gridX - 1)) / gridX) + (kio * ((((c1_dimension + (gridX - 1)) / gridX - 0 / gridX) + 1) / 2) + (0 / gridX) / 2)));
    Point<2> bEnd = Point<2>(TACO_MIN(((iln * ((((aPartitionBounds0hi - aPartitionBounds0lo) + 1) + 1) / 2) + ((((aPartitionBounds0hi - aPartitionBounds0lo) + 1) + 1) / 2 - 1)) + aPartitionBounds0lo),bDomain.hi()[0]), TACO_MIN((ko * ((c1_dimension + (gridX - 1)) / gridX) + (kio * ((((c1_dimension + (gridX - 1)) / gridX - 0 / gridX) + 1) / 2) + (((c1_dimension + (gridX - 1)) / gridX + 1) / 2 - 1))),bDomain.hi()[1]));
    Rect<2> bRect = Rect<2>(bStart, bEnd);
    if (!bDomain.contains(bRect.lo) || !bDomain.contains(bRect.hi)) {
      bRect = bRect.make_empty();
    }
    bColoring[(*itr)] = bRect;
    Point<2> cStart = Point<2>((ko * ((c1_dimension + (gridX - 1)) / gridX) + (kio * ((((c1_dimension + (gridX - 1)) / gridX - 0 / gridX) + 1) / 2) + (0 / gridX) / 2)), (jln * ((((aPartitionBounds1hi - aPartitionBounds1lo) + 1) + 1) / 2) + aPartitionBounds1lo));
    Point<2> cEnd = Point<2>(TACO_MIN((ko * ((c1_dimension + (gridX - 1)) / gridX) + (kio * ((((c1_dimension + (gridX - 1)) / gridX - 0 / gridX) + 1) / 2) + (((c1_dimension + (gridX - 1)) / gridX + 1) / 2 - 1))),cDomain.hi()[0]), TACO_MIN(((jln * ((((aPartitionBounds1hi - aPartitionBounds1lo) + 1) + 1) / 2) + ((((aPartitionBounds1hi - aPartitionBounds1lo) + 1) + 1) / 2 - 1)) + aPartitionBounds1lo),cDomain.hi()[1]));
    Rect<2> cRect = Rect<2>(cStart, cEnd);
    if (!cDomain.contains(cRect.lo) || !cDomain.contains(cRect.hi)) {
      cRect = cRect.make_empty();
    }
    cColoring[(*itr)] = cRect;
  }
  auto bPartition = runtime->create_index_partition(ctx, b_index_space, domain, bColoring, LEGION_DISJOINT_COMPLETE_KIND);
  auto cPartition = runtime->create_index_partition(ctx, c_index_space, domain, cColoring, LEGION_DISJOINT_COMPLETE_KIND);
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t kio = (*itr);
    RegionRequirement aReq = RegionRequirement(get_logical_region(a), READ_WRITE, EXCLUSIVE, get_logical_region(a));
    aReq.add_field(FID_VAL);
    auto bsubReg = runtime->get_logical_subregion_by_color(ctx, runtime->get_logical_partition(ctx, get_logical_region(b), bPartition), kio);
    RegionRequirement bReq = RegionRequirement(bsubReg, READ_ONLY, EXCLUSIVE, get_logical_region(b));
    bReq.add_field(FID_VAL);
    auto csubReg = runtime->get_logical_subregion_by_color(ctx, runtime->get_logical_partition(ctx, get_logical_region(c), cPartition), kio);
    RegionRequirement cReq = RegionRequirement(csubReg, READ_ONLY, EXCLUSIVE, get_logical_region(c));
    cReq.add_field(FID_VAL);
    task_7Args taskArgsRaw;
    taskArgsRaw.kio = kio;
    TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_7Args));
    TaskLauncher launcher = TaskLauncher(taskID(7), taskArgs);
    launcher.add_region_requirement(aReq);
    launcher.add_region_requirement(bReq);
    launcher.add_region_requirement(cReq);
    launcher.tag = launcher.tag | TACOMapper::UNTRACK_VALID_REGIONS;
    runtime->execute_task(ctx, launcher);
  }

}

void task_9(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a = regions[0];
  PhysicalRegion b = regions[1];
  PhysicalRegion c = regions[2];

  task_9Args* args = (task_9Args*)(task->args);
  int64_t aPartitionBounds0hi = args->aPartitionBounds0hi;
  int64_t aPartitionBounds0lo = args->aPartitionBounds0lo;
  int64_t aPartitionBounds1hi = args->aPartitionBounds1hi;
  int64_t aPartitionBounds1lo = args->aPartitionBounds1lo;
  int32_t c1_dimension = args->c1_dimension;
  int32_t gridX = args->gridX;
  int32_t ko = args->ko;

  auto a_index_space = get_index_space(a);
  auto b_index_space = get_index_space(b);
  auto c_index_space = get_index_space(c);

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>(1, 1);
  auto distFused1IndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFused1IndexSpace));
  auto bDomain = runtime->get_index_space_domain(ctx, b_index_space);
  auto cDomain = runtime->get_index_space_domain(ctx, c_index_space);
  auto aDomain = runtime->get_index_space_domain(ctx, a_index_space);
  DomainPointColoring bColoring = DomainPointColoring();
  DomainPointColoring cColoring = DomainPointColoring();
  DomainPointColoring aColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int32_t iln = (*itr)[0];
    int32_t jln = (*itr)[1];
    Point<2> bStart = Point<2>((iln * ((((aPartitionBounds0hi - aPartitionBounds0lo) + 1) + 1) / 2) + aPartitionBounds0lo), (ko * ((c1_dimension + (gridX - 1)) / gridX) + 0 / gridX));
    Point<2> bEnd = Point<2>(TACO_MIN(((iln * ((((aPartitionBounds0hi - aPartitionBounds0lo) + 1) + 1) / 2) + ((((aPartitionBounds0hi - aPartitionBounds0lo) + 1) + 1) / 2 - 1)) + aPartitionBounds0lo),bDomain.hi()[0]), TACO_MIN((ko * ((c1_dimension + (gridX - 1)) / gridX) + ((c1_dimension + (gridX - 1)) / gridX - 1)),bDomain.hi()[1]));
    Rect<2> bRect = Rect<2>(bStart, bEnd);
    if (!bDomain.contains(bRect.lo) || !bDomain.contains(bRect.hi)) {
      bRect = bRect.make_empty();
    }
    bColoring[(*itr)] = bRect;
    Point<2> cStart = Point<2>((ko * ((c1_dimension + (gridX - 1)) / gridX) + 0 / gridX), (jln * ((((aPartitionBounds1hi - aPartitionBounds1lo) + 1) + 1) / 2) + aPartitionBounds1lo));
    Point<2> cEnd = Point<2>(TACO_MIN((ko * ((c1_dimension + (gridX - 1)) / gridX) + ((c1_dimension + (gridX - 1)) / gridX - 1)),cDomain.hi()[0]), TACO_MIN(((jln * ((((aPartitionBounds1hi - aPartitionBounds1lo) + 1) + 1) / 2) + ((((aPartitionBounds1hi - aPartitionBounds1lo) + 1) + 1) / 2 - 1)) + aPartitionBounds1lo),cDomain.hi()[1]));
    Rect<2> cRect = Rect<2>(cStart, cEnd);
    if (!cDomain.contains(cRect.lo) || !cDomain.contains(cRect.hi)) {
      cRect = cRect.make_empty();
    }
    cColoring[(*itr)] = cRect;
    Point<2> aStart = Point<2>((iln * ((((aPartitionBounds0hi - aPartitionBounds0lo) + 1) + 1) / 2) + aPartitionBounds0lo), (jln * ((((aPartitionBounds1hi - aPartitionBounds1lo) + 1) + 1) / 2) + aPartitionBounds1lo));
    Point<2> aEnd = Point<2>(TACO_MIN(((iln * ((((aPartitionBounds0hi - aPartitionBounds0lo) + 1) + 1) / 2) + ((((aPartitionBounds0hi - aPartitionBounds0lo) + 1) + 1) / 2 - 1)) + aPartitionBounds0lo),aDomain.hi()[0]), TACO_MIN(((jln * ((((aPartitionBounds1hi - aPartitionBounds1lo) + 1) + 1) / 2) + ((((aPartitionBounds1hi - aPartitionBounds1lo) + 1) + 1) / 2 - 1)) + aPartitionBounds1lo),aDomain.hi()[1]));
    Rect<2> aRect = Rect<2>(aStart, aEnd);
    if (!aDomain.contains(aRect.lo) || !aDomain.contains(aRect.hi)) {
      aRect = aRect.make_empty();
    }
    aColoring[(*itr)] = aRect;
  }
  auto bPartition = runtime->create_index_partition(ctx, b_index_space, domain, bColoring, LEGION_ALIASED_COMPLETE_KIND);
  auto cPartition = runtime->create_index_partition(ctx, c_index_space, domain, cColoring, LEGION_ALIASED_COMPLETE_KIND);
  auto aPartition = runtime->create_index_partition(ctx, a_index_space, domain, aColoring, LEGION_DISJOINT_COMPLETE_KIND);
  LogicalPartition aLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(a), aPartition);
  RegionRequirement aReq = RegionRequirement(aLogicalPartition, 0, READ_WRITE, EXCLUSIVE, get_logical_region(a));
  aReq.add_field(FID_VAL);
  aReq.tag = Mapping::DefaultMapper::VIRTUAL_MAP;
  LogicalPartition bLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(b), bPartition);
  RegionRequirement bReq = RegionRequirement(bLogicalPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(b));
  bReq.add_field(FID_VAL);
  bReq.tag = Mapping::DefaultMapper::VIRTUAL_MAP;
  LogicalPartition cLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(c), cPartition);
  RegionRequirement cReq = RegionRequirement(cLogicalPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(c));
  cReq.add_field(FID_VAL);
  cReq.tag = Mapping::DefaultMapper::VIRTUAL_MAP;
  task_8Args taskArgsRaw;
  taskArgsRaw.aPartitionBounds0hi = aPartitionBounds0hi;
  taskArgsRaw.aPartitionBounds0lo = aPartitionBounds0lo;
  taskArgsRaw.aPartitionBounds1hi = aPartitionBounds1hi;
  taskArgsRaw.aPartitionBounds1lo = aPartitionBounds1lo;
  taskArgsRaw.c1_dimension = c1_dimension;
  taskArgsRaw.gridX = gridX;
  taskArgsRaw.ko = ko;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_8Args));
  IndexLauncher launcher = IndexLauncher(taskID(8), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(aReq);
  launcher.add_region_requirement(bReq);
  launcher.add_region_requirement(cReq);
  launcher.tag = launcher.tag | Mapping::DefaultMapper::SAME_ADDRESS_SPACE;
  runtime->execute_index_space(ctx, launcher);

}

void task_10(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a = regions[0];
  PhysicalRegion b = regions[1];
  PhysicalRegion c = regions[2];

  int32_t distFused = task->index_point[0];
  task_10Args* args = (task_10Args*)(task->args);
  int32_t c1_dimension = args->c1_dimension;
  int32_t gridX = args->gridX;

  auto a_index_space = get_index_space(a);
  auto b_index_space = get_index_space(b);
  auto c_index_space = get_index_space(c);

  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
  auto aPartitionBounds = runtime->get_index_space_domain(ctx, a_index_space);
  int64_t aPartitionBounds0lo = aPartitionBounds.lo()[0];
  int64_t aPartitionBounds0hi = aPartitionBounds.hi()[0];
  int64_t aPartitionBounds1lo = aPartitionBounds.lo()[1];
  int64_t aPartitionBounds1hi = aPartitionBounds.hi()[1];
  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>((gridX - 1));
  auto koIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(koIndexSpace));
  auto bDomain = runtime->get_index_space_domain(ctx, b_index_space);
  auto cDomain = runtime->get_index_space_domain(ctx, c_index_space);
  DomainPointColoring bColoring = DomainPointColoring();
  DomainPointColoring cColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t ko = (*itr)[0];
    Point<2> bStart = Point<2>(aPartitionBounds0lo, (ko * ((c1_dimension + (gridX - 1)) / gridX) + 0 / gridX));
    Point<2> bEnd = Point<2>(TACO_MIN(((((aPartitionBounds0hi - aPartitionBounds0lo) + 1) - 1) + aPartitionBounds0lo),bDomain.hi()[0]), TACO_MIN((ko * ((c1_dimension + (gridX - 1)) / gridX) + ((c1_dimension + (gridX - 1)) / gridX - 1)),bDomain.hi()[1]));
    Rect<2> bRect = Rect<2>(bStart, bEnd);
    if (!bDomain.contains(bRect.lo) || !bDomain.contains(bRect.hi)) {
      bRect = bRect.make_empty();
    }
    bColoring[(*itr)] = bRect;
    Point<2> cStart = Point<2>((ko * ((c1_dimension + (gridX - 1)) / gridX) + 0 / gridX), aPartitionBounds1lo);
    Point<2> cEnd = Point<2>(TACO_MIN((ko * ((c1_dimension + (gridX - 1)) / gridX) + ((c1_dimension + (gridX - 1)) / gridX - 1)),cDomain.hi()[0]), TACO_MIN(((((aPartitionBounds1hi - aPartitionBounds1lo) + 1) - 1) + aPartitionBounds1lo),cDomain.hi()[1]));
    Rect<2> cRect = Rect<2>(cStart, cEnd);
    if (!cDomain.contains(cRect.lo) || !cDomain.contains(cRect.hi)) {
      cRect = cRect.make_empty();
    }
    cColoring[(*itr)] = cRect;
  }
  auto bPartition = runtime->create_index_partition(ctx, b_index_space, domain, bColoring, LEGION_DISJOINT_COMPLETE_KIND);
  auto cPartition = runtime->create_index_partition(ctx, c_index_space, domain, cColoring, LEGION_DISJOINT_COMPLETE_KIND);
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t ko = (*itr);
    RegionRequirement aReq = RegionRequirement(get_logical_region(a), READ_WRITE, EXCLUSIVE, get_logical_region(a));
    aReq.add_field(FID_VAL);
    aReq.tag = Mapping::DefaultMapper::VIRTUAL_MAP;
    auto bsubReg = runtime->get_logical_subregion_by_color(ctx, runtime->get_logical_partition(ctx, get_logical_region(b), bPartition), ko);
    RegionRequirement bReq = RegionRequirement(bsubReg, READ_ONLY, EXCLUSIVE, get_logical_region(b));
    bReq.add_field(FID_VAL);
    bReq.tag = Mapping::DefaultMapper::VIRTUAL_MAP;
    auto csubReg = runtime->get_logical_subregion_by_color(ctx, runtime->get_logical_partition(ctx, get_logical_region(c), cPartition), ko);
    RegionRequirement cReq = RegionRequirement(csubReg, READ_ONLY, EXCLUSIVE, get_logical_region(c));
    cReq.add_field(FID_VAL);
    cReq.tag = Mapping::DefaultMapper::VIRTUAL_MAP;
    task_9Args taskArgsRaw;
    taskArgsRaw.aPartitionBounds0hi = aPartitionBounds0hi;
    taskArgsRaw.aPartitionBounds0lo = aPartitionBounds0lo;
    taskArgsRaw.aPartitionBounds1hi = aPartitionBounds1hi;
    taskArgsRaw.aPartitionBounds1lo = aPartitionBounds1lo;
    taskArgsRaw.c1_dimension = c1_dimension;
    taskArgsRaw.gridX = gridX;
    taskArgsRaw.ko = ko;
    TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_9Args));
    TaskLauncher launcher = TaskLauncher(taskID(9), taskArgs);
    launcher.add_region_requirement(aReq);
    launcher.add_region_requirement(bReq);
    launcher.add_region_requirement(cReq);
    runtime->execute_task(ctx, launcher);
  }

}

void computeLegion(Context ctx, Runtime* runtime, LogicalRegion a, LogicalRegion b, LogicalRegion c, LogicalPartition aPartition, int32_t gridX) {
  auto b_index_space = get_index_space(b);
  int c1_dimension = runtime->get_index_space_domain(get_index_space(c)).hi()[0] + 1;
  auto c_index_space = get_index_space(c);

  DomainT<2> domain = runtime->get_index_partition_color_space(ctx, get_index_partition(aPartition));
  auto bDomain = runtime->get_index_space_domain(ctx, b_index_space);
  auto cDomain = runtime->get_index_space_domain(ctx, c_index_space);
  DomainPointColoring bColoring = DomainPointColoring();
  DomainPointColoring cColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    DomainPoint domPoint = (*itr);
    auto aPartitionBounds = runtime->get_index_space_domain(runtime->get_logical_subregion_by_color(ctx, aPartition, domPoint).get_index_space());
    int64_t aPartitionBounds0lo = aPartitionBounds.lo()[0];
    int64_t aPartitionBounds0hi = aPartitionBounds.hi()[0];
    int64_t aPartitionBounds1lo = aPartitionBounds.lo()[1];
    int64_t aPartitionBounds1hi = aPartitionBounds.hi()[1];
    Point<2> bStart = Point<2>(aPartitionBounds0lo, (0 / gridX));
    Point<2> bEnd = Point<2>(TACO_MIN(((((aPartitionBounds0hi - aPartitionBounds0lo) + 1) - 1) + aPartitionBounds0lo),bDomain.hi()[0]), TACO_MIN(((gridX - 1) * ((c1_dimension + (gridX - 1)) / gridX) + ((c1_dimension + (gridX - 1)) / gridX - 1)),bDomain.hi()[1]));
    Rect<2> bRect = Rect<2>(bStart, bEnd);
    if (!bDomain.contains(bRect.lo) || !bDomain.contains(bRect.hi)) {
      bRect = bRect.make_empty();
    }
    bColoring[(*itr)] = bRect;
    Point<2> cStart = Point<2>((0 / gridX), aPartitionBounds1lo);
    Point<2> cEnd = Point<2>(TACO_MIN(((gridX - 1) * ((c1_dimension + (gridX - 1)) / gridX) + ((c1_dimension + (gridX - 1)) / gridX - 1)),cDomain.hi()[0]), TACO_MIN(((((aPartitionBounds1hi - aPartitionBounds1lo) + 1) - 1) + aPartitionBounds1lo),cDomain.hi()[1]));
    Rect<2> cRect = Rect<2>(cStart, cEnd);
    if (!cDomain.contains(cRect.lo) || !cDomain.contains(cRect.hi)) {
      cRect = cRect.make_empty();
    }
    cColoring[(*itr)] = cRect;
  }
  auto bPartition = runtime->create_index_partition(ctx, b_index_space, domain, bColoring, LEGION_ALIASED_COMPLETE_KIND);
  auto cPartition = runtime->create_index_partition(ctx, c_index_space, domain, cColoring, LEGION_ALIASED_COMPLETE_KIND);
  RegionRequirement aReq = RegionRequirement(aPartition, 0, READ_WRITE, EXCLUSIVE, get_logical_region(a));
  aReq.add_field(FID_VAL);
  aReq.tag = Mapping::DefaultMapper::VIRTUAL_MAP;
  LogicalPartition bLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(b), bPartition);
  RegionRequirement bReq = RegionRequirement(bLogicalPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(b));
  bReq.add_field(FID_VAL);
  bReq.tag = Mapping::DefaultMapper::VIRTUAL_MAP;
  LogicalPartition cLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(c), cPartition);
  RegionRequirement cReq = RegionRequirement(cLogicalPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(c));
  cReq.add_field(FID_VAL);
  cReq.tag = Mapping::DefaultMapper::VIRTUAL_MAP;
  task_10Args taskArgsRaw;
  taskArgsRaw.c1_dimension = c1_dimension;
  taskArgsRaw.gridX = gridX;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_10Args));
  IndexLauncher launcher = IndexLauncher(taskID(10), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(aReq);
  launcher.add_region_requirement(bReq);
  launcher.add_region_requirement(cReq);
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
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_inner();
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
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_inner();
    Runtime::preregister_task_variant<task_4>(registrar, "task_4");
  }
  {
    TaskVariantRegistrar registrar(taskID(5), "task_5");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_5>(registrar, "task_5");
  }
  {
    TaskVariantRegistrar registrar(taskID(6), "task_6");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_inner();
    Runtime::preregister_task_variant<task_6>(registrar, "task_6");
  }
  {
    TaskVariantRegistrar registrar(taskID(7), "task_7");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<task_7>(registrar, "task_7");
  }
  {
    TaskVariantRegistrar registrar(taskID(8), "task_8");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_inner();
    Runtime::preregister_task_variant<task_8>(registrar, "task_8");
  }
  {
    TaskVariantRegistrar registrar(taskID(9), "task_9");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_inner();
    Runtime::preregister_task_variant<task_9>(registrar, "task_9");
  }
  {
    TaskVariantRegistrar registrar(taskID(10), "task_10");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_inner();
    Runtime::preregister_task_variant<task_10>(registrar, "task_10");
  }
}
