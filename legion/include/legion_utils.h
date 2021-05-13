#ifndef TACO_LEGION_UTILS_H
#define TACO_LEGION_UTILS_H
#include <functional>

#include "legion.h"
#include "taco_legion_header.h"

template<typename T>
void allocate_tensor_fields(Legion::Context ctx, Legion::Runtime* runtime, Legion::FieldSpace valSpace) {
  Legion::FieldAllocator allocator = runtime->create_field_allocator(ctx, valSpace);
  allocator.allocate_field(sizeof(T), FID_VAL);
  runtime->attach_name(valSpace, FID_VAL, "vals");
}

Legion::PhysicalRegion getRegionToWrite(Legion::Context ctx, Legion::Runtime* runtime, Legion::LogicalRegion r, Legion::LogicalRegion parent);

const int TACO_FILL_TASK = 1;
template<typename T>
void tacoFill(Legion::Context ctx, Legion::Runtime* runtime, Legion::LogicalRegion r, T val) {
  Legion::TaskLauncher l(TACO_FILL_TASK, Legion::TaskArgument(&val, sizeof(T)));
  l.add_region_requirement(Legion::RegionRequirement(r, WRITE_ONLY, EXCLUSIVE, r).add_field(FID_VAL));
  runtime->execute_task(ctx, l);
}

template<typename T>
void tacoFillTask(const Legion::Task* task, const std::vector<Legion::PhysicalRegion>& regions, Legion::Context ctx, Legion::Runtime* runtime) {
  Legion::PhysicalRegion r = regions[0];
  auto ispace = r.get_logical_region().get_index_space();
  switch (ispace.get_dim()) {
#define BLOCK(DIM) \
        case DIM:  \
          {        \
            typedef Legion::FieldAccessor<WRITE_ONLY,T,DIM,Legion::coord_t,Realm::AffineAccessor<T,DIM,Legion::coord_t>> Accessor; \
            Accessor ar(r, FID_VAL);                                                                                             \
            for (Legion::PointInRectIterator<DIM> itr(runtime->get_index_space_domain(ispace)); itr(); itr++) {                 \
              ar[*itr] = *(T*)(task->args);       \
            }       \
            break; \
          }
    LEGION_FOREACH_N(BLOCK)
#undef BLOCK
    default:
      assert(false);
  }
}

void benchmark(std::function<void(void)> f);

#define TACO_MAIN(FillType) \
  int main(int argc, char **argv) { \
    int TID_TOP_LEVEL = 1000;       \
    Runtime::set_top_level_task_id(TID_TOP_LEVEL); \
    {               \
      TaskVariantRegistrar registrar(TID_TOP_LEVEL, "top_level"); \
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC)); \
      Runtime::preregister_task_variant<top_level_task>(registrar, "top_level"); \
    }               \
    {               \
      TaskVariantRegistrar registrar(TACO_FILL_TASK, "taco_fill"); \
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC)); \
      Runtime::preregister_task_variant<tacoFillTask<FillType>>(registrar, "taco_fill"); \
    }               \
    Runtime::add_registration_callback(register_taco_mapper);     \
    registerTacoTasks();            \
    return Runtime::start(argc, argv);             \
  }
#endif //TACO_LEGION_UTILS_H
