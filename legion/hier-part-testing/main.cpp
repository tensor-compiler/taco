#include "legion.h"

using namespace Legion;

enum TaskIDs {
  TID_TOP_LEVEL,
  TID_PART_LEVEL_1,
  TID_COMPUTE_LEVEL_1,
  TID_COMPUTE_LEVEL_2,
};
enum FieldIDs {
  FID_VAL,
};

enum PartColors {
  TEST_COLOR,
};

void partLevel1(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  auto cspace = runtime->create_index_space(ctx, Rect<2>{{0, 0}, {1, 1}});
  Transform<2,2> transform;
  transform[0][0] = 1;
  transform[0][1] = 0;
  transform[1][0] = 0;
  transform[1][1] = 1;
  auto lo = regions[0].get_bounds<2, coord_t>().bounds.lo;
  auto bounds = Domain(lo, lo);
  auto part = runtime->create_partition_by_restriction(ctx, regions[0].get_logical_region().get_index_space(), cspace, transform, bounds, DISJOINT_COMPLETE_KIND, TEST_COLOR);
  auto lp = runtime->get_logical_partition(ctx, regions[0].get_logical_region(), part);
//  std::cout << "Task index point: " << task->index_point << std::endl;
//  for (PointInDomainIterator<2> itr(runtime->get_index_space_domain(cspace)); itr(); itr++) {
//    auto subreg = runtime->get_logical_subregion_by_color(ctx, lp, *itr);
//    std::cout << *itr << " -> " << runtime->get_index_space_domain(subreg.get_index_space()) << std::endl;
//  }
}

LogicalPartition partition(Context ctx, Runtime* runtime, LogicalRegion r) {
  auto cspace = runtime->create_index_space(ctx, Rect<2>{{0, 0}, {1, 1}});
  Transform<2,2> transform;
  transform[0][0] = 2;
  transform[0][1] = 0;
  transform[1][0] = 0;
  transform[1][1] = 2;
  auto bounds = Rect<2>({0, 0}, {1, 1});
  auto part = runtime->create_partition_by_restriction(ctx, r.get_index_space(), cspace, transform, bounds, DISJOINT_COMPLETE_KIND, TEST_COLOR);
  auto lp = runtime->get_logical_partition(ctx, r, part);
//  for (PointInDomainIterator<2> itr(runtime->get_index_space_domain(cspace)); itr(); itr++) {
//    auto subreg = runtime->get_logical_subregion_by_color(ctx, lp, *itr);
//    std::cout << *itr << " -> " << runtime->get_index_space_domain(subreg.get_index_space()) << std::endl;
//  }
  {
    IndexTaskLauncher launcher(TID_PART_LEVEL_1, runtime->get_index_space_domain(cspace), TaskArgument(), ArgumentMap());
    launcher.add_region_requirement(RegionRequirement(lp, 0, READ_ONLY, EXCLUSIVE, r).add_field(FID_VAL));
    runtime->execute_index_space(ctx, launcher).wait_all_results();
  }
  return lp;
}

void compLevel2(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  auto bounds = regions[0].get_bounds<2, coord_t>();
  std::cout << bounds.bounds << std::endl;
}

void compLevel1(const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx, Runtime* runtime) {
  auto subreg = regions[0].get_logical_region();
  auto lp = runtime->get_logical_partition_by_color(ctx, subreg, TEST_COLOR);
  IndexTaskLauncher launcher(TID_COMPUTE_LEVEL_2, runtime->get_index_partition_color_space(lp.get_index_partition()), TaskArgument(), ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(lp, 0, READ_ONLY, EXCLUSIVE, subreg).add_field(FID_VAL));
  runtime->execute_index_space(ctx, launcher);
}

void compute(Context ctx, Runtime* runtime, LogicalRegion r, LogicalPartition p) {
  IndexTaskLauncher launcher(TID_COMPUTE_LEVEL_1, runtime->get_index_partition_color_space(p.get_index_partition()), TaskArgument(), ArgumentMap());
  launcher.add_region_requirement(RegionRequirement(p, 0, READ_ONLY, EXCLUSIVE, r).add_field(FID_VAL));
  runtime->execute_index_space(ctx, launcher).wait_all_results();
}

void top_level_task(const Task* task, const std::vector<PhysicalRegion>&, Context ctx, Runtime* runtime) {
  auto fspace = runtime->create_field_space(ctx);
  {
    Legion::FieldAllocator alloc = runtime->create_field_allocator(ctx, fspace);
    alloc.allocate_field(sizeof(double), FID_VAL);
  }
  auto n = 4;
  auto ispace = runtime->create_index_space(ctx, Rect<2>{{0, 0}, {n - 1, n - 1}});
  auto r = runtime->create_logical_region(ctx, ispace, fspace);
  runtime->fill_field(ctx, r, r, FID_VAL, (double)(0));

  auto part = partition(ctx, runtime, r);
  compute(ctx, runtime, r, part);
}

int main(int argc, char** argv) {
  Runtime::set_top_level_task_id(TID_TOP_LEVEL);
  {
    TaskVariantRegistrar registrar(TID_TOP_LEVEL, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }
  {
    TaskVariantRegistrar registrar(TID_PART_LEVEL_1, "partLevel1");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<partLevel1>(registrar, "partLevel1");
  }
  {
    TaskVariantRegistrar registrar(TID_COMPUTE_LEVEL_1, "compLevel1");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<compLevel1>(registrar, "compLevel1");
  }
  {
    TaskVariantRegistrar registrar(TID_COMPUTE_LEVEL_2, "compLevel2");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<compLevel2>(registrar, "compLevel2");
  }
  return Runtime::start(argc, argv);
}
