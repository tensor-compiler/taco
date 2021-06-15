#include "taco_legion_header.h"
#include "taco_mapper.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
using namespace Legion;
typedef FieldAccessor<READ_ONLY,int32_t,2,coord_t,Realm::AffineAccessor<int32_t,2,coord_t>> AccessorROint32_t2;
typedef ReductionAccessor<SumReduction<int32_t>,true,2,coord_t,Realm::AffineAccessor<int32_t,2,coord_t>> AccessorReduceint32_t2;

struct task_1Args {
  int32_t sfID;
};
struct task_2Args {
  int32_t sfID;
};
struct task_3Args {
  int32_t sfID;
};
struct task_4Args {
  int32_t b1_dimension;
  int32_t c1_dimension;
  int32_t c2_dimension;
  int32_t in;
  int32_t jn;
  int32_t k1s;
  int32_t kn;
};
struct task_5Args {
  int32_t b1_dimension;
  int32_t c1_dimension;
  int32_t c2_dimension;
};

void task_1(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a = regions[0];

  int32_t distFused = task->index_point[0];

  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
  int32_t kn = getIndexPoint(task, 2);
}

LogicalPartition placeLegionA(Context ctx, Runtime* runtime, LogicalRegion a) {
  int a1_dimension = runtime->get_index_space_domain(get_index_space(a)).hi()[0] + 1;
  int a2_dimension = runtime->get_index_space_domain(get_index_space(a)).hi()[1] + 1;
  auto a_index_space = get_index_space(a);

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>(4, 4, 0);
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  auto aDomain = runtime->get_index_space_domain(ctx, a_index_space);
  DomainPointColoring aColoring = DomainPointColoring();
  for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    Point<2> aStart = Point<2>((in * ((a1_dimension + 4) / 5)), (jn * ((a2_dimension + 4) / 5)));
    Point<2> aEnd = Point<2>(TACO_MIN((in * ((a1_dimension + 4) / 5) + ((a1_dimension + 4) / 5 - 1)), aDomain.hi()[0]), TACO_MIN((jn * ((a2_dimension + 4) / 5) + ((a2_dimension + 4) / 5 - 1)), aDomain.hi()[1]));
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
  dims.push_back(5);
  dims.push_back(5);
  dims.push_back(2);
  registerPlacementShardingFunctor(ctx, runtime, shardingID(3), dims);
  task_1Args taskArgsRaw;
  taskArgsRaw.sfID = shardingID(3);
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

  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
  int32_t kn = getIndexPoint(task, 2);
}

LogicalPartition placeLegionB(Context ctx, Runtime* runtime, LogicalRegion b) {
  int b1_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[0] + 1;
  int b2_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[1] + 1;
  auto b_index_space = get_index_space(b);

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>(4, 4, 0);
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  auto bDomain = runtime->get_index_space_domain(ctx, b_index_space);
  DomainPointColoring bColoring = DomainPointColoring();
  for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    Point<2> bStart = Point<2>((in * ((b1_dimension + 4) / 5)), (jn * ((b2_dimension + 4) / 5)));
    Point<2> bEnd = Point<2>(TACO_MIN((in * ((b1_dimension + 4) / 5) + ((b1_dimension + 4) / 5 - 1)), bDomain.hi()[0]), TACO_MIN((jn * ((b2_dimension + 4) / 5) + ((b2_dimension + 4) / 5 - 1)), bDomain.hi()[1]));
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
  dims.push_back(5);
  dims.push_back(5);
  dims.push_back(2);
  registerPlacementShardingFunctor(ctx, runtime, shardingID(4), dims);
  task_2Args taskArgsRaw;
  taskArgsRaw.sfID = shardingID(4);
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

  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
  int32_t kn = getIndexPoint(task, 2);
}

LogicalPartition placeLegionC(Context ctx, Runtime* runtime, LogicalRegion c) {
  int c1_dimension = runtime->get_index_space_domain(get_index_space(c)).hi()[0] + 1;
  int c2_dimension = runtime->get_index_space_domain(get_index_space(c)).hi()[1] + 1;
  auto c_index_space = get_index_space(c);

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>(4, 4, 0);
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<3>(lowerBound, upperBound));
  DomainT<3> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<3>(distFusedIndexSpace));
  auto cDomain = runtime->get_index_space_domain(ctx, c_index_space);
  DomainPointColoring cColoring = DomainPointColoring();
  for (PointInDomainIterator<3> itr = PointInDomainIterator<3>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    Point<2> cStart = Point<2>((in * ((c1_dimension + 4) / 5)), (jn * ((c2_dimension + 4) / 5)));
    Point<2> cEnd = Point<2>(TACO_MIN((in * ((c1_dimension + 4) / 5) + ((c1_dimension + 4) / 5 - 1)), cDomain.hi()[0]), TACO_MIN((jn * ((c2_dimension + 4) / 5) + ((c2_dimension + 4) / 5 - 1)), cDomain.hi()[1]));
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
  dims.push_back(5);
  dims.push_back(5);
  dims.push_back(2);
  registerPlacementShardingFunctor(ctx, runtime, shardingID(5), dims);
  task_3Args taskArgsRaw;
  taskArgsRaw.sfID = shardingID(5);
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

  task_4Args* args = (task_4Args*)(task->args);
  int32_t b1_dimension = args->b1_dimension;
  int32_t c1_dimension = args->c1_dimension;
  int32_t c2_dimension = args->c2_dimension;
  int32_t in = args->in;
  int32_t jn = args->jn;
  int32_t k1s = args->k1s;
  int32_t kn = args->kn;

  AccessorROint32_t2 b_vals(b, FID_VAL);
  AccessorROint32_t2 c_vals(c, FID_VAL);
  AccessorReduceint32_t2 a_vals(a, FID_VAL, LEGION_REDOP_SUM_INT32);

  int32_t k1 = (jn + (in + k1s)) % 2;
  for (int32_t il = 0; il < ((b1_dimension + 4) / 5); il++) {
    int32_t i = in * ((b1_dimension + 4) / 5) + il;
    if (i >= b1_dimension)
      continue;

    if (i >= (in + 1) * ((b1_dimension + 4) / 5))
      continue;

    for (int32_t jl = 0; jl < ((c2_dimension + 4) / 5); jl++) {
      int32_t j = jn * ((c2_dimension + 4) / 5) + jl;
      Point<2> a_access_point = Point<2>(i, j);
      if (j >= c2_dimension)
        continue;

      if (j >= (jn + 1) * ((c2_dimension + 4) / 5))
        continue;

      for (int32_t k2 = 0; k2 < (((c1_dimension + 1) / 2 + 1) / 2); k2++) {
        int32_t kl = k1 * (((c1_dimension + 1) / 2 + 1) / 2) + k2;
        if (kl >= (k1 + 1) * (((c1_dimension + 1) / 2 + 1) / 2))
          continue;

        int32_t k = kn * ((c1_dimension + 1) / 2) + kl;
        Point<2> b_access_point = Point<2>(i, k);
        Point<2> c_access_point = Point<2>(k, j);
        if (k >= c1_dimension)
          continue;

        if (k >= (kn + 1) * ((c1_dimension + 1) / 2))
          continue;

        a_vals[a_access_point] <<= b_vals[b_access_point] * c_vals[c_access_point];
      }
    }
  }
}

void task_5(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a = regions[0];
  PhysicalRegion b = regions[1];
  PhysicalRegion c = regions[2];

  int32_t distFused = task->index_point[0];
  task_5Args* args = (task_5Args*)(task->args);
  int32_t b1_dimension = args->b1_dimension;
  int32_t c1_dimension = args->c1_dimension;
  int32_t c2_dimension = args->c2_dimension;

  auto b_index_space = get_index_space(b);
  auto c_index_space = get_index_space(c);

  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
  int32_t kn = getIndexPoint(task, 2);
  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>(1);
  auto k1sIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(k1sIndexSpace));
  auto bDomain = runtime->get_index_space_domain(ctx, b_index_space);
  auto cDomain = runtime->get_index_space_domain(ctx, c_index_space);
  DomainPointColoring bColoring = DomainPointColoring();
  DomainPointColoring cColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t k1s = (*itr)[0];
    Point<2> bStart = Point<2>((in * ((b1_dimension + 4) / 5)), (kn * ((c1_dimension + 1) / 2) + ((jn + (in + k1s)) % 2) * (((c1_dimension + 1) / 2 + 1) / 2)));
    Point<2> bEnd = Point<2>(TACO_MIN((in * ((b1_dimension + 4) / 5) + ((b1_dimension + 4) / 5 - 1)), bDomain.hi()[0]), TACO_MIN((kn * ((c1_dimension + 1) / 2) + (((jn + (in + k1s)) % 2) * (((c1_dimension + 1) / 2 + 1) / 2) + (((c1_dimension + 1) / 2 + 1) / 2 - 1))), bDomain.hi()[1]));
    Rect<2> bRect = Rect<2>(bStart, bEnd);
    if (!bDomain.contains(bRect.lo) || !bDomain.contains(bRect.hi)) {
      bRect = bRect.make_empty();
    }
    bColoring[(*itr)] = bRect;
    Point<2> cStart = Point<2>((kn * ((c1_dimension + 1) / 2) + ((jn + (in + k1s)) % 2) * (((c1_dimension + 1) / 2 + 1) / 2)), (jn * ((c2_dimension + 4) / 5)));
    Point<2> cEnd = Point<2>(TACO_MIN((kn * ((c1_dimension + 1) / 2) + (((jn + (in + k1s)) % 2) * (((c1_dimension + 1) / 2 + 1) / 2) + (((c1_dimension + 1) / 2 + 1) / 2 - 1))), cDomain.hi()[0]), TACO_MIN((jn * ((c2_dimension + 4) / 5) + ((c2_dimension + 4) / 5 - 1)), cDomain.hi()[1]));
    Rect<2> cRect = Rect<2>(cStart, cEnd);
    if (!cDomain.contains(cRect.lo) || !cDomain.contains(cRect.hi)) {
      cRect = cRect.make_empty();
    }
    cColoring[(*itr)] = cRect;
  }
  auto bPartition = runtime->create_index_partition(ctx, b_index_space, domain, bColoring, LEGION_DISJOINT_COMPLETE_KIND);
  auto cPartition = runtime->create_index_partition(ctx, c_index_space, domain, cColoring, LEGION_DISJOINT_COMPLETE_KIND);
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t k1s = (*itr);
    RegionRequirement aReq = RegionRequirement(get_logical_region(a), LEGION_REDOP_SUM_INT32, LEGION_SIMULTANEOUS, get_logical_region(a));
    aReq.add_field(FID_VAL);
    auto bsubReg = runtime->get_logical_subregion_by_color(ctx, runtime->get_logical_partition(ctx, get_logical_region(b), bPartition), k1s);
    RegionRequirement bReq = RegionRequirement(bsubReg, READ_ONLY, EXCLUSIVE, get_logical_region(b));
    bReq.add_field(FID_VAL);
    auto csubReg = runtime->get_logical_subregion_by_color(ctx, runtime->get_logical_partition(ctx, get_logical_region(c), cPartition), k1s);
    RegionRequirement cReq = RegionRequirement(csubReg, READ_ONLY, EXCLUSIVE, get_logical_region(c));
    cReq.add_field(FID_VAL);
    task_4Args taskArgsRaw;
    taskArgsRaw.b1_dimension = b1_dimension;
    taskArgsRaw.c1_dimension = c1_dimension;
    taskArgsRaw.c2_dimension = c2_dimension;
    taskArgsRaw.in = in;
    taskArgsRaw.jn = jn;
    taskArgsRaw.k1s = k1s;
    taskArgsRaw.kn = kn;
    TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_4Args));
    TaskLauncher launcher = TaskLauncher(taskID(4), taskArgs);
    launcher.add_region_requirement(aReq);
    launcher.add_region_requirement(bReq);
    launcher.add_region_requirement(cReq);
    runtime->execute_task(ctx, launcher);
  }

}

void computeLegion(Context ctx, Runtime* runtime, LogicalRegion a, LogicalRegion b, LogicalRegion c) {
  auto a_index_space = get_index_space(a);
  int b1_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[0] + 1;
  auto b_index_space = get_index_space(b);
  int c1_dimension = runtime->get_index_space_domain(get_index_space(c)).hi()[0] + 1;
  int c2_dimension = runtime->get_index_space_domain(get_index_space(c)).hi()[1] + 1;
  auto c_index_space = get_index_space(c);

  Point<3> lowerBound = Point<3>(0, 0, 0);
  Point<3> upperBound = Point<3>(4, 4, 1);
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
    Point<2> aStart = Point<2>((in * ((b1_dimension + 4) / 5)), (jn * ((c2_dimension + 4) / 5)));
    Point<2> aEnd = Point<2>(TACO_MIN((in * ((b1_dimension + 4) / 5) + ((b1_dimension + 4) / 5 - 1)), aDomain.hi()[0]), TACO_MIN((jn * ((c2_dimension + 4) / 5) + ((c2_dimension + 4) / 5 - 1)), aDomain.hi()[1]));
    Rect<2> aRect = Rect<2>(aStart, aEnd);
    if (!aDomain.contains(aRect.lo) || !aDomain.contains(aRect.hi)) {
      aRect = aRect.make_empty();
    }
    aColoring[(*itr)] = aRect;
    Point<2> bStart = Point<2>((in * ((b1_dimension + 4) / 5)), (kn * ((c1_dimension + 1) / 2)));
    Point<2> bEnd = Point<2>(TACO_MIN((in * ((b1_dimension + 4) / 5) + ((b1_dimension + 4) / 5 - 1)), bDomain.hi()[0]), TACO_MIN((kn * ((c1_dimension + 1) / 2) + ((c1_dimension + 1) / 2 - 1)), bDomain.hi()[1]));
    Rect<2> bRect = Rect<2>(bStart, bEnd);
    if (!bDomain.contains(bRect.lo) || !bDomain.contains(bRect.hi)) {
      bRect = bRect.make_empty();
    }
    bColoring[(*itr)] = bRect;
    Point<2> cStart = Point<2>((kn * ((c1_dimension + 1) / 2)), (jn * ((c2_dimension + 4) / 5)));
    Point<2> cEnd = Point<2>(TACO_MIN((kn * ((c1_dimension + 1) / 2) + ((c1_dimension + 1) / 2 - 1)), cDomain.hi()[0]), TACO_MIN((jn * ((c2_dimension + 4) / 5) + ((c2_dimension + 4) / 5 - 1)), cDomain.hi()[1]));
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
  RegionRequirement aReq = RegionRequirement(aLogicalPartition, 0, LEGION_REDOP_SUM_INT32, LEGION_SIMULTANEOUS, get_logical_region(a));
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
  task_5Args taskArgsRaw;
  taskArgsRaw.b1_dimension = b1_dimension;
  taskArgsRaw.c1_dimension = c1_dimension;
  taskArgsRaw.c2_dimension = c2_dimension;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_5Args));
  IndexLauncher launcher = IndexLauncher(taskID(5), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(aReq);
  launcher.add_region_requirement(bReq);
  launcher.add_region_requirement(cReq);
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
