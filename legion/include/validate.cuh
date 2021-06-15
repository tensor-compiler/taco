#ifndef TACO_LG_VALIDATE_CUH
#define TACO_LG_VALIDATE_CUH

#include "task_ids.h"
#include "legion.h"
#include "pitches.h"
#include "taco_legion_header.h"
#include "validate.h"

const int THREADS_PER_BLOCK = 256;

template<int DIM, typename T>
__global__
void tacoValidateGPUKernel(
    Legion::FieldAccessor <READ_ONLY, T, DIM, Legion::coord_t, Realm::AffineAccessor<T, DIM, Legion::coord_t>> a,
    T value, Pitches<DIM - 1> pitches, Legion::Point<DIM> lo, size_t volume) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= volume) return;
  assert(a[pitches.unflatten(idx, lo)] == value);
}

template<int DIM, typename T>
void tacoValidateGPU(const Legion::Task* task, Legion::PhysicalRegion r, Legion::Rect<DIM> rect) {
  typedef Legion::FieldAccessor<READ_ONLY,T,DIM,Legion::coord_t,Realm::AffineAccessor<T,DIM,Legion::coord_t>> Accessor;
  Accessor ar(r, FID_VAL);
  Pitches<DIM - 1> pitches;
  auto volume = pitches.flatten(rect);
  auto blocks = (volume + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  tacoValidateGPUKernel<DIM, T><<<blocks, THREADS_PER_BLOCK>>>(ar, *(T*)(task->args), pitches, rect.lo, volume);
}

template<typename T>
void tacoValidateGPUTask(const Legion::Task* task, const std::vector<Legion::PhysicalRegion>& regions, Legion::Context ctx, Legion::Runtime* runtime) {
  Legion::PhysicalRegion r = regions[0];
  auto ispace = r.get_logical_region().get_index_space();
  auto domain = runtime->get_index_space_domain(ispace);
  switch (ispace.get_dim()) {
#define BLOCK(DIM) \
        case DIM:  \
          {        \
            tacoValidateGPU<DIM, T>(task, r, domain); \
            break; \
          }
    LEGION_FOREACH_N(BLOCK)
#undef BLOCK
    default:
      assert(false);
  }
}

template <typename T>
void registerGPUValidateTask() {
  {
    Legion::TaskVariantRegistrar registrar(TID_TACO_VALIDATE_TASK, "taco_validate");
    registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::TOC_PROC));
    Legion::Runtime::preregister_task_variant<tacoValidateGPUTask<T>>(registrar, "taco_validate");
  }
}

#endif // TACO_LG_VALIDATE_CUH
