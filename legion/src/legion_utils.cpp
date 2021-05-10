#include <functional>
#include <chrono>
#include <iostream>

#include "legion.h"
#include "legion_utils.h"

using namespace Legion;

Legion::PhysicalRegion getRegionToWrite(Legion::Context ctx, Legion::Runtime* runtime, Legion::LogicalRegion r, Legion::LogicalRegion parent) {
  Legion::RegionRequirement req(r, READ_WRITE, EXCLUSIVE, parent);
  req.add_field(FID_VAL);
  return runtime->map_region(ctx, req);
}

void benchmark(std::function<void(void)> f) {
  auto start = std::chrono::high_resolution_clock::now();
  f();
  auto end = std::chrono::high_resolution_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "Execution time: " << ms << " ms." << std::endl;
}

// TODO (rohany): Need to have a version of this for each type? I would like to dispatch
//  on the type of the region.
void tacoFillTask(const Legion::Task* task, const std::vector<Legion::PhysicalRegion>& regions, Legion::Context ctx, Legion::Runtime* runtime) {
  PhysicalRegion r = regions[0];
  auto ispace = r.get_logical_region().get_index_space();
  switch (ispace.get_dim()) {
#define BLOCK(DIM) \
        case DIM:  \
          {        \
            typedef FieldAccessor<WRITE_ONLY,int32_t,DIM,coord_t, Realm::AffineAccessor<int32_t,DIM,coord_t>> Accessor; \
            Accessor ar(r, FID_VAL);                                                                                             \
            for (PointInRectIterator<DIM> itr(runtime->get_index_space_domain(ispace)); itr(); itr++) {                 \
              ar[*itr] = *(int*)(task->args);       \
            }       \
            break; \
          }
    LEGION_FOREACH_N(BLOCK)
#undef BLOCK
    default:
      assert(false);
  }
}

void tacoFill(Legion::Context ctx, Legion::Runtime* runtime, Legion::LogicalRegion r, int val) {
  TaskLauncher l(TACO_FILL_TASK, TaskArgument(&val, sizeof(val)));
  l.add_region_requirement(RegionRequirement(r, WRITE_ONLY, EXCLUSIVE, r).add_field(FID_VAL));
  runtime->execute_task(ctx, l);
}
