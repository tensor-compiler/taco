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
// Variant of benchmark that collects the runtime into a vector.
void benchmark(Legion::Context ctx, Legion::Runtime* runtime, std::vector<size_t>& times, std::function<void(void)> f);
// Variant of benchmark that uses Legion's async timing infrastructure.
void benchmarkAsyncCall(Legion::Context ctx, Legion::Runtime* runtime, std::vector<size_t>& times, std::function<void(void)> f);

// Utility function to get the number of flops performed by various
// tensor and matrix operations.
size_t getGEMMFLOPCount(size_t M, size_t N, size_t K);
size_t getTTMCFLOPCount(size_t I, size_t J, size_t K, size_t L);
size_t getMTTKRPFLOPCount(size_t I, size_t J, size_t K, size_t L);

// Utility function to do the unit conversions for GFLOPS.
double getGFLOPS(size_t flopCount, size_t ms);

// Utility function to return the average of a list of numbers.
// TODO (rohany): We can update this to ignore the first element, or the maximum
//  and minimum element etc.
template<typename T>
T average(std::vector<T> vals) {
  T sum = 0;
  for (auto elem : vals) {
    sum += elem;
  }
  return sum / (T(vals.size()));
}

// We forward declare these functions. If we are building with CUDA, then
// the CUDA files define them. Otherwise, the CPP files define them.
void initCuBLAS(Legion::Context ctx, Legion::Runtime* runtime);
void initCUDA();

#define TACO_MAIN(FillType) \
  int main(int argc, char **argv) { \
    Runtime::set_top_level_task_id(TID_TOP_LEVEL); \
    {               \
      TaskVariantRegistrar registrar(TID_TOP_LEVEL, "top_level"); \
      registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC)); \
      registrar.set_replicable();   \
      Runtime::preregister_task_variant<top_level_task>(registrar, "top_level"); \
    }                       \
    if (TACO_FEATURE_OPENMP) {               \
      TaskVariantRegistrar registrar(TID_TOP_LEVEL, "top_level"); \
      registrar.add_constraint(ProcessorConstraint(Processor::OMP_PROC)); \
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

