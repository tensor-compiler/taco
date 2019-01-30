#include "taco/taco_tensor_t.h"
#include "taco/cuda.h"
#include <cstdio>
#include <cstdlib>

void * alloc_mem(size_t size);
void free_mem(void *ptr);

// Allocates from unified memory or using malloc depending on what memory is being used
void * alloc_mem(size_t size) {
  if (taco::should_use_CUDA_unified_memory()) {
    return taco::cuda_unified_alloc(size);
  }
  else {
    return malloc(size);
  }
}

// Free from unified memory or using free depending on what memory is being used
void free_mem(void *ptr) {
  if (taco::should_use_CUDA_unified_memory()) {
    taco::cuda_unified_free(ptr);
  }
  else {
    free(ptr);
  }
}

taco_tensor_t* init_taco_tensor_t(int32_t order, int32_t csize,
                        int32_t* dimensions, int32_t* modeOrdering,
                        taco_mode_t* mode_types) {
  taco_tensor_t* t = (taco_tensor_t *) alloc_mem(sizeof(taco_tensor_t));
  t->order         = order;
  t->dimensions = (int32_t *) alloc_mem(order * sizeof(int32_t));
  t->mode_ordering = (int32_t *) alloc_mem(order * sizeof(int32_t));
  t->mode_types = (taco_mode_t *) alloc_mem(order * sizeof(taco_mode_t));
  t->indices = (uint8_t ***) alloc_mem(order * sizeof(uint8_t***));
  t->csize         = csize;

  for (int32_t i = 0; i < order; i++) {
    t->dimensions[i]    = dimensions[i];
    t->mode_ordering[i] = modeOrdering[i];
    t->mode_types[i]    = mode_types[i];
    switch (t->mode_types[i]) {
      case taco_mode_dense:
        t->indices[i] = (uint8_t **) alloc_mem(1 * sizeof(uint8_t **));
        break;
      case taco_mode_sparse:
        t->indices[i] = (uint8_t **) alloc_mem(2 * sizeof(uint8_t **));
        break;
    }
  }
  return t;
}

void deinit_taco_tensor_t(taco_tensor_t* t) {
  for (int i = 0; i < t->order; i++) {
    free_mem(t->indices[i]);
  }
  free_mem(t->indices);

  free_mem(t->dimensions);
  free_mem(t->mode_ordering);
  free_mem(t->mode_types);
  free_mem(t);
}