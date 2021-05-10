#ifndef TACO_LEGION_UTILS_H
#define TACO_LEGION_UTILS_H

#include "legion.h"
#include "taco_legion_header.h"

template<typename T>
void allocate_tensor_fields(Legion::Context ctx, Legion::Runtime* runtime, Legion::FieldSpace valSpace) {
  Legion::FieldAllocator allocator = runtime->create_field_allocator(ctx, valSpace);
  allocator.allocate_field(sizeof(T), FID_VAL);
  runtime->attach_name(valSpace, FID_VAL, "vals");
}

Legion::PhysicalRegion getRegionToWrite(Legion::Context ctx, Legion::Runtime* runtime, Legion::LogicalRegion r, Legion::LogicalRegion parent) {
  Legion::RegionRequirement req(r, READ_WRITE, EXCLUSIVE, parent);
  req.add_field(FID_VAL);
  return runtime->map_region(ctx, req);
}

#define TACO_MAIN() \
  int main(int argc, char **argv) { \
    int TID_TOP_LEVEL = 1000;       \
    Runtime::set_top_level_task_id(TID_TOP_LEVEL); \
    {               \
      TaskVariantRegistrar registrar(TID_TOP_LEVEL, "top_level"); \
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC)); \
      Runtime::preregister_task_variant<top_level_task>(registrar, "top_level"); \
    }               \
    Runtime::add_registration_callback(register_taco_mapper);     \
    registerTacoTasks();            \
    return Runtime::start(argc, argv);             \
  }

#endif //TACO_LEGION_UTILS_H
