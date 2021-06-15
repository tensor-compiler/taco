#ifndef TACO_LEGION_UTILS_H
#define TACO_LEGION_UTILS_H
#include <functional>

#include "legion.h"
#include "task_ids.h"
#include "taco_legion_header.h"
#include "mappers/default_mapper.h"
#include "taco_mapper.h"
#include "taco/version.h"
#include "fill.h"
#include "validate.h"

#ifdef TACO_USE_CUDA
#include "cudalibs.h"
#endif

template<typename T>
void allocate_tensor_fields(Legion::Context ctx, Legion::Runtime* runtime, Legion::FieldSpace valSpace) {
  Legion::FieldAllocator allocator = runtime->create_field_allocator(ctx, valSpace);
  allocator.allocate_field(sizeof(T), FID_VAL);
  runtime->attach_name(valSpace, FID_VAL, "vals");
}

Legion::PhysicalRegion getRegionToWrite(Legion::Context ctx, Legion::Runtime* runtime, Legion::LogicalRegion r, Legion::LogicalRegion parent);

// Benchmarking utility functions.
void benchmark(std::function<void(void)> f);
// Variant of benchmark that prints only once in a control replicated setting.
void benchmark(Legion::Context ctx, Legion::Runtime* runtime, std::function<void(void)> f);

// We forward declare these functions. If we are building with CUDA, then
// the CUDA files define them. Otherwise, the CPP files define them.
void initCuBLAS(Legion::Context ctx, Legion::Runtime* runtime);
void initCUDA();

#define TACO_MAIN(FillType) \
  int main(int argc, char **argv) { \
    Runtime::set_top_level_task_id(TID_TOP_LEVEL); \
    {               \
      TaskVariantRegistrar registrar(TID_TOP_LEVEL, "top_level"); \
      if (TACO_FEATURE_OPENMP) {    \
        registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC)); \
      } else {              \
        registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC)); \
      }                     \
      registrar.set_replicable();   \
      Runtime::preregister_task_variant<top_level_task>(registrar, "top_level"); \
    }                       \
    registerTACOFillTasks<FillType>();             \
    registerTACOValidateTasks<FillType>();             \
    Runtime::add_registration_callback(register_taco_mapper);     \
    initCUDA(); \
    registerTacoTasks();    \
    Runtime::preregister_sharding_functor(TACOShardingFunctorID, new TACOShardingFunctor()); \
    return Runtime::start(argc, argv);             \
  }
#endif //TACO_LEGION_UTILS_H

