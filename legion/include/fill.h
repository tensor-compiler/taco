#ifndef TACO_LG_FILL_H
#define TACO_LG_FILL_H

#include "task_ids.h"
#include "legion.h"
#include "pitches.h"
#include "taco_legion_header.h"
#include "taco/version.h"

template<int DIM, typename T>
void tacoFillCPU(const Legion::Task* task, Legion::PhysicalRegion r, Legion::Rect<DIM> rect) {
  typedef Legion::FieldAccessor<WRITE_ONLY,T,DIM,Legion::coord_t,Realm::AffineAccessor<T,DIM,Legion::coord_t>> Accessor;
  Accessor ar(r, FID_VAL);
  Pitches<DIM - 1> pitches;
  auto volume = pitches.flatten(rect);
  for (size_t i = 0; i < volume; i++) {
    ar[pitches.unflatten(i, rect.lo)] = *(T*)(task->args);
  }
}

template<int DIM, typename T>
void tacoFillOMP(const Legion::Task* task, Legion::PhysicalRegion r, Legion::Rect<DIM> rect) {
  typedef Legion::FieldAccessor<WRITE_ONLY,T,DIM,Legion::coord_t,Realm::AffineAccessor<T,DIM,Legion::coord_t>> Accessor;
  Accessor ar(r, FID_VAL);
  Pitches<DIM - 1> pitches;
  auto volume = pitches.flatten(rect);
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < volume; i++) {
    ar[pitches.unflatten(i, rect.lo)] = *(T*)(task->args);
  }
}

template<typename T>
void tacoFillCPUTask(const Legion::Task* task, const std::vector<Legion::PhysicalRegion>& regions, Legion::Context ctx, Legion::Runtime* runtime) {
  Legion::PhysicalRegion r = regions[0];
  auto ispace = r.get_logical_region().get_index_space();
  auto domain = runtime->get_index_space_domain(ispace);
  switch (ispace.get_dim()) {
#define BLOCK(DIM) \
        case DIM:  \
          {        \
            tacoFillCPU<DIM, T>(task, r, domain); \
            break; \
          }
    LEGION_FOREACH_N(BLOCK)
#undef BLOCK
    default:
      assert(false);
  }
}

template<typename T>
void tacoFillOMPTask(const Legion::Task* task, const std::vector<Legion::PhysicalRegion>& regions, Legion::Context ctx, Legion::Runtime* runtime) {
  Legion::PhysicalRegion r = regions[0];
  auto ispace = r.get_logical_region().get_index_space();
  auto domain = runtime->get_index_space_domain(ispace);
  switch (ispace.get_dim()) {
#define BLOCK(DIM) \
        case DIM:  \
          {        \
            tacoFillOMP<DIM, T>(task, r, domain); \
            break; \
          }
    LEGION_FOREACH_N(BLOCK)
#undef BLOCK
    default:
      assert(false);
  }
}

template<typename T>
void tacoFill(Legion::Context ctx, Legion::Runtime* runtime, Legion::LogicalRegion r, T val) {
  size_t pieces = 0;
  // Favor TOC proc > OMP proc > CPU proc. The default mapper performs this same heuristic
  // as well, so there's nothing more we need to do.
  auto numGPU = runtime->select_tunable_value(ctx, Legion::Mapping::DefaultMapper::DEFAULT_TUNABLE_GLOBAL_GPUS).get<size_t>();
  auto numOMP = runtime->select_tunable_value(ctx, Legion::Mapping::DefaultMapper::DEFAULT_TUNABLE_GLOBAL_OMPS).get<size_t>();
  auto numCPU = runtime->select_tunable_value(ctx, Legion::Mapping::DefaultMapper::DEFAULT_TUNABLE_GLOBAL_CPUS).get<size_t>();
  if (numGPU != 0) {
    pieces = numGPU;
  } else if (numOMP != 0) {
    pieces = numOMP;
  } else if (numCPU != 0) {
    pieces = numCPU;
  } else {
    assert(false);
  }
  auto ispace = runtime->create_index_space(ctx, Legion::Rect<1>(0, pieces - 1));
  auto ipart = runtime->create_equal_partition(ctx, r.get_index_space(), ispace);
  auto lpart = runtime->get_logical_partition(ctx, r, ipart);
  Legion::IndexLauncher l(TID_TACO_FILL_TASK, runtime->get_index_space_domain(ispace), Legion::TaskArgument(&val, sizeof(T)), Legion::ArgumentMap());
  l.add_region_requirement(Legion::RegionRequirement(lpart, 0, WRITE_ONLY, EXCLUSIVE, r).add_field(FID_VAL));
  runtime->execute_index_space(ctx, l);
}

template<typename T>
void tacoFill(Legion::Context ctx, Legion::Runtime* runtime, Legion::LogicalRegion r, Legion::LogicalPartition part, T val) {
  auto domain = runtime->get_index_partition_color_space(ctx, get_index_partition(part));
  Legion::IndexLauncher l(TID_TACO_FILL_TASK, domain, Legion::TaskArgument(&val, sizeof(T)), Legion::ArgumentMap());
  l.add_region_requirement(Legion::RegionRequirement(part, 0, WRITE_ONLY, EXCLUSIVE, r).add_field(FID_VAL));
  runtime->execute_index_space(ctx, l);
}

// If we're building with CUDA, then declare the fill kernel.
#ifdef TACO_USE_CUDA
template<typename T>
void registerGPUFillTask();
#endif

template <typename T>
void registerTACOFillTasks() {
  // Register the CPU variant.
  {
    Legion::TaskVariantRegistrar registrar(TID_TACO_FILL_TASK, "taco_fill");
    registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
    Legion::Runtime::preregister_task_variant<tacoFillCPUTask<T>>(registrar, "taco_fill");
  }
  // Register the OMP variant if present.
  if (TACO_FEATURE_OPENMP) {
    Legion::TaskVariantRegistrar registrar(TID_TACO_FILL_TASK, "taco_fill");
    registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::OMP_PROC));
    Legion::Runtime::preregister_task_variant<tacoFillOMPTask<T>>(registrar, "taco_fill");
  }
  // Register a CUDA variant if we have a CUDA build.
#ifdef TACO_USE_CUDA
  registerGPUFillTask<T>();
#endif
}

#endif // TACO_LG_FILL_H
