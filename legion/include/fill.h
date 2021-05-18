#ifndef TACO_LG_FILL_H
#define TACO_LG_FILL_H

#include "legion.h"
#include "pitches.h"
#include "taco/version.h"

#ifdef TACO_USE_CUDA
#include "fill.cuh"
#endif

const int TACO_FILL_TASK = 1;

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
  // Favor OMP proc > CPU proc. The default mapper performs this same heuristic
  // as well, so there's nothing more we need to do.
  auto numOMP = runtime->select_tunable_value(ctx, Legion::Mapping::DefaultMapper::DEFAULT_TUNABLE_GLOBAL_OMPS).get<size_t>();
  auto numCPU = runtime->select_tunable_value(ctx, Legion::Mapping::DefaultMapper::DEFAULT_TUNABLE_GLOBAL_CPUS).get<size_t>();
  if (numOMP != 0) {
    pieces = numOMP;
  } else if (numCPU != 0) {
    pieces = numCPU;
  } else {
    assert(false);
  }
  auto ispace = runtime->create_index_space(ctx, pieces);
  auto ipart = runtime->create_equal_partition(ctx, r.get_index_space(), ispace);
  auto lpart = runtime->get_logical_partition(ctx, r, ipart);
  Legion::IndexLauncher l(TACO_FILL_TASK, runtime->get_index_space_domain(ispace), Legion::TaskArgument(&val, sizeof(T)), Legion::ArgumentMap());
  l.add_region_requirement(Legion::RegionRequirement(lpart, 0, WRITE_ONLY, EXCLUSIVE, r).add_field(FID_VAL));
  runtime->execute_index_space(ctx, l);
}

template <typename T>
void registerTACOFillTasks() {
  // Register the CPU variant.
  {
    Legion::TaskVariantRegistrar registrar(TACO_FILL_TASK, "taco_fill");
    registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::LOC_PROC));
    Legion::Runtime::preregister_task_variant<tacoFillCPUTask<T>>(registrar, "taco_fill");
  }
  // Register the OMP variant if present.
  if (TACO_FEATURE_OPENMP) {
    Legion::TaskVariantRegistrar registrar(TACO_FILL_TASK, "taco_fill");
    registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::OMP_PROC));
    Legion::Runtime::preregister_task_variant<tacoFillOMPTask<T>>(registrar, "taco_fill");
  }
  // Register a CUDA variant if we have a CUDA build.
#ifdef TACO_USE_CUDA
  {
    Legion::TaskVariantRegistrar registrar(TACO_FILL_TASK, "taco_fill");
    registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::TOC_PROC));
    Legion::Runtime::preregister_task_variant<tacoFillGPUTask<T>>(registrar, "taco_fill");
  }
#endif
}

#endif // TACO_LG_FILL_H