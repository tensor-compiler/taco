#include "taco_legion_header.h"
#include "taco_mapper.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
using namespace Legion;
typedef FieldAccessor<READ_ONLY,int32_t,2,coord_t,Realm::AffineAccessor<int32_t,2,coord_t>> AccessorROint32_t2;
typedef FieldAccessor<READ_WRITE,int32_t,2,coord_t,Realm::AffineAccessor<int32_t,2,coord_t>> AccessorRWint32_t2;

struct task_1Args {
};
struct task_2Args {
};
struct task_3Args {
};
struct task_4Args {
  int32_t ko;
  int32_t b1_dimension;
  int32_t c2_dimension;
  int32_t c1_dimension;
};
struct task_5Args {
  int32_t b1_dimension;
  int32_t b2_dimension;
  int32_t c2_dimension;
  int32_t c1_dimension;
};

void task_1(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a = regions[0];

  int32_t distFused = task->index_point[0];

  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
}

LogicalPartition placeLegionA(Context ctx, Runtime* runtime, LogicalRegion a) {
  int a1_dimension = runtime->get_index_space_domain(get_index_space(a)).hi()[0] + 1;
  int a2_dimension = runtime->get_index_space_domain(get_index_space(a)).hi()[1] + 1;
  auto a_index_space = get_index_space(a);

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>(1, 1);
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  DomainPointColoring aColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    Point<2> aStart = Point<2>((in * ((a1_dimension + 1) / 2)), (jn * ((a2_dimension + 1) / 2)));
    Point<2> aEnd = Point<2>(TACO_MIN((in * ((a1_dimension + 1) / 2) + ((a1_dimension + 1) / 2 - 1)),(a1_dimension - 1)), TACO_MIN((jn * ((a2_dimension + 1) / 2) + ((a2_dimension + 1) / 2 - 1)),(a2_dimension - 1)));
    Rect<2> aRect = Rect<2>(aStart, aEnd);
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
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  return runtime->get_logical_partition(ctx, get_logical_region(a), aPartition);

}

void task_2(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion b = regions[0];

  int32_t distFused = task->index_point[0];

  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
}

LogicalPartition placeLegionB(Context ctx, Runtime* runtime, LogicalRegion b) {
  int b1_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[0] + 1;
  int b2_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[1] + 1;
  auto b_index_space = get_index_space(b);

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>(1, 1);
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  DomainPointColoring bColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    Point<2> bStart = Point<2>((in * ((b1_dimension + 1) / 2)), (jn * ((b2_dimension + 1) / 2)));
    Point<2> bEnd = Point<2>(TACO_MIN((in * ((b1_dimension + 1) / 2) + ((b1_dimension + 1) / 2 - 1)),(b1_dimension - 1)), TACO_MIN((jn * ((b2_dimension + 1) / 2) + ((b2_dimension + 1) / 2 - 1)),(b2_dimension - 1)));
    Rect<2> bRect = Rect<2>(bStart, bEnd);
    bColoring[(*itr)] = bRect;
  }
  auto bPartition = runtime->create_index_partition(ctx, b_index_space, domain, bColoring, LEGION_COMPUTE_KIND);
  LogicalPartition bLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(b), bPartition);
  RegionRequirement bReq = RegionRequirement(bLogicalPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(b));
  bReq.add_field(FID_VAL);
  task_2Args taskArgsRaw;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_2Args));
  IndexLauncher launcher = IndexLauncher(taskID(2), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(bReq);
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  return runtime->get_logical_partition(ctx, get_logical_region(b), bPartition);

}

void task_3(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion c = regions[0];

  int32_t distFused = task->index_point[0];

  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
}

LogicalPartition placeLegionC(Context ctx, Runtime* runtime, LogicalRegion c) {
  int c1_dimension = runtime->get_index_space_domain(get_index_space(c)).hi()[0] + 1;
  int c2_dimension = runtime->get_index_space_domain(get_index_space(c)).hi()[1] + 1;
  auto c_index_space = get_index_space(c);

  Point<2> lowerBound = Point<2>(0, 0);
  Point<2> upperBound = Point<2>(1, 1);
  auto distFusedIndexSpace = runtime->create_index_space(ctx, Rect<2>(lowerBound, upperBound));
  DomainT<2> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<2>(distFusedIndexSpace));
  DomainPointColoring cColoring = DomainPointColoring();
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    int32_t in = (*itr)[0];
    int32_t jn = (*itr)[1];
    Point<2> cStart = Point<2>((in * ((c1_dimension + 1) / 2)), (jn * ((c2_dimension + 1) / 2)));
    Point<2> cEnd = Point<2>(TACO_MIN((in * ((c1_dimension + 1) / 2) + ((c1_dimension + 1) / 2 - 1)),(c1_dimension - 1)), TACO_MIN((jn * ((c2_dimension + 1) / 2) + ((c2_dimension + 1) / 2 - 1)),(c2_dimension - 1)));
    Rect<2> cRect = Rect<2>(cStart, cEnd);
    cColoring[(*itr)] = cRect;
  }
  auto cPartition = runtime->create_index_partition(ctx, c_index_space, domain, cColoring, LEGION_COMPUTE_KIND);
  LogicalPartition cLogicalPartition = runtime->get_logical_partition(ctx, get_logical_region(c), cPartition);
  RegionRequirement cReq = RegionRequirement(cLogicalPartition, 0, READ_ONLY, EXCLUSIVE, get_logical_region(c));
  cReq.add_field(FID_VAL);
  task_3Args taskArgsRaw;
  TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_3Args));
  IndexLauncher launcher = IndexLauncher(taskID(3), domain, taskArgs, ArgumentMap());
  launcher.add_region_requirement(cReq);
  auto fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  return runtime->get_logical_partition(ctx, get_logical_region(c), cPartition);

}

void task_4(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  PhysicalRegion a = regions[0];
  PhysicalRegion b = regions[1];
  PhysicalRegion c = regions[2];

  task_4Args* args = (task_4Args*)(task->args);
  int32_t ko = args->ko;
  int32_t b1_dimension = args->b1_dimension;
  int32_t c2_dimension = args->c2_dimension;
  int32_t c1_dimension = args->c1_dimension;

  auto a_index_space = get_index_space(a);
  AccessorROint32_t2 c_vals(c, FID_VAL);
  AccessorROint32_t2 b_vals(b, FID_VAL);
  AccessorRWint32_t2 a_vals(a, FID_VAL);

  auto aPartitionBounds = runtime->get_index_space_domain(ctx, a_index_space);
  for (int32_t il = aPartitionBounds.lo()[0]; il < (aPartitionBounds.hi()[0] + 1); il++) {
    if (il >= b1_dimension)
      continue;

    for (int32_t jl = aPartitionBounds.lo()[1]; jl < (aPartitionBounds.hi()[1] + 1); jl++) {
      Point<2> a_access_point = Point<2>(il, jl);
      if (jl >= c2_dimension)
        continue;

      for (int32_t ki = 0; ki < 256; ki++) {
        int32_t k = ko * 256 + ki;
        Point<2> b_access_point = Point<2>(il, k);
        Point<2> c_access_point = Point<2>(k, jl);
        if (k >= c1_dimension)
          continue;

        a_vals[a_access_point] = a_vals[a_access_point] + b_vals[b_access_point] * c_vals[c_access_point];
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
  int32_t b2_dimension = args->b2_dimension;
  int32_t c2_dimension = args->c2_dimension;
  int32_t c1_dimension = args->c1_dimension;

  auto c_index_space = get_index_space(c);
  auto b_index_space = get_index_space(b);
  auto a_index_space = get_index_space(a);

  int32_t in = getIndexPoint(task, 0);
  int32_t jn = getIndexPoint(task, 1);
  auto aPartitionBounds = runtime->get_index_space_domain(ctx, a_index_space);
  Point<1> lowerBound = Point<1>(0);
  Point<1> upperBound = Point<1>(((c1_dimension + 255) / 256 - 1));
  auto koIndexSpace = runtime->create_index_space(ctx, Rect<1>(lowerBound, upperBound));
  DomainT<1> domain = runtime->get_index_space_domain(ctx, IndexSpaceT<1>(koIndexSpace));
  DomainPointColoring bColoring = DomainPointColoring();
  DomainPointColoring cColoring = DomainPointColoring();
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t ko = (*itr)[0];
    Point<2> bStart = Point<2>(aPartitionBounds.lo()[0], (ko * 256));
    Point<2> bEnd = Point<2>(TACO_MIN(((aPartitionBounds.hi()[0] + 1) - 1),(b1_dimension - 1)), TACO_MIN((ko * 256 + 255),(b2_dimension - 1)));
    Rect<2> bRect = Rect<2>(bStart, bEnd);
    bColoring[(*itr)] = bRect;
    Point<2> cStart = Point<2>((ko * 256), aPartitionBounds.lo()[1]);
    Point<2> cEnd = Point<2>(TACO_MIN((ko * 256 + 255),(c1_dimension - 1)), TACO_MIN(((aPartitionBounds.hi()[1] + 1) - 1),(c2_dimension - 1)));
    Rect<2> cRect = Rect<2>(cStart, cEnd);
    cColoring[(*itr)] = cRect;
  }
  auto bPartition = runtime->create_index_partition(ctx, b_index_space, domain, bColoring, LEGION_DISJOINT_KIND);
  auto cPartition = runtime->create_index_partition(ctx, c_index_space, domain, cColoring, LEGION_DISJOINT_KIND);
  for (PointInDomainIterator<1> itr = PointInDomainIterator<1>(domain); itr.valid(); itr++) {
    int32_t ko = (*itr);
    RegionRequirement aReq = RegionRequirement(get_logical_region(a), READ_WRITE, EXCLUSIVE, get_logical_region(a));
    aReq.add_field(FID_VAL);
    auto bsubReg = runtime->get_logical_subregion_by_color(ctx, runtime->get_logical_partition(ctx, get_logical_region(b), bPartition), ko);
    RegionRequirement bReq = RegionRequirement(bsubReg, READ_ONLY, EXCLUSIVE, get_logical_region(b));
    bReq.add_field(FID_VAL);
    auto csubReg = runtime->get_logical_subregion_by_color(ctx, runtime->get_logical_partition(ctx, get_logical_region(c), cPartition), ko);
    RegionRequirement cReq = RegionRequirement(csubReg, READ_ONLY, EXCLUSIVE, get_logical_region(c));
    cReq.add_field(FID_VAL);
    task_4Args taskArgsRaw;
    taskArgsRaw.ko = ko;
    taskArgsRaw.b1_dimension = b1_dimension;
    taskArgsRaw.c2_dimension = c2_dimension;
    taskArgsRaw.c1_dimension = c1_dimension;
    TaskArgument taskArgs = TaskArgument(&taskArgsRaw, sizeof(task_4Args));
    TaskLauncher launcher = TaskLauncher(taskID(4), taskArgs);
    launcher.add_region_requirement(aReq);
    launcher.add_region_requirement(bReq);
    launcher.add_region_requirement(cReq);
    runtime->execute_task(ctx, launcher);
  }

}

void computeLegion(Context ctx, Runtime* runtime, LogicalRegion a, LogicalRegion b, LogicalRegion c, LogicalPartition aPartition) {
  int b1_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[0] + 1;
  int b2_dimension = runtime->get_index_space_domain(get_index_space(b)).hi()[1] + 1;
  int c1_dimension = runtime->get_index_space_domain(get_index_space(c)).hi()[0] + 1;
  int c2_dimension = runtime->get_index_space_domain(get_index_space(c)).hi()[1] + 1;

  DomainT<2> domain = runtime->get_index_partition_color_space(ctx, get_index_partition(aPartition));
  for (PointInDomainIterator<2> itr = PointInDomainIterator<2>(domain); itr.valid(); itr++) {
    DomainPoint domPoint = (*itr);
    auto aPartitionBounds = runtime->get_index_space_domain(runtime->get_logical_subregion_by_color(ctx, aPartition, domPoint).get_index_space());
  }
  RegionRequirement aReq = RegionRequirement(aPartition, 0, READ_WRITE, EXCLUSIVE, get_logical_region(a));
  aReq.add_field(FID_VAL);
  aReq.tag = Mapping::DefaultMapper::VIRTUAL_MAP;
  RegionRequirement bReq = RegionRequirement(get_logical_region(b), READ_ONLY, EXCLUSIVE, get_logical_region(b));
  bReq.add_field(FID_VAL);
  bReq.tag = Mapping::DefaultMapper::VIRTUAL_MAP;
  RegionRequirement cReq = RegionRequirement(get_logical_region(c), READ_ONLY, EXCLUSIVE, get_logical_region(c));
  cReq.add_field(FID_VAL);
  cReq.tag = Mapping::DefaultMapper::VIRTUAL_MAP;
  task_5Args taskArgsRaw;
  taskArgsRaw.b1_dimension = b1_dimension;
  taskArgsRaw.b2_dimension = b2_dimension;
  taskArgsRaw.c2_dimension = c2_dimension;
  taskArgsRaw.c1_dimension = c1_dimension;
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
    Runtime::preregister_task_variant<task_5>(registrar, "task_5");
  }
}
