/// This file defines the runtime struct used to pass raw tensors to generated
/// code.  Note: this file must be valid C99, not C++.
/// This *must* be kept in sync with the version used in codegen_c.cpp

#ifndef TACO_TENSOR_T_DEFINED
#define TACO_TENSOR_T_DEFINED

typedef enum { taco_mode_dense, taco_mode_sparse } taco_mode_t;

typedef struct taco_tensor_t {
  int32_t      order;         // tensor order (number of modes)
  int32_t*     dimensions;    // tensor dimensions
  int32_t      csize;         // component size
  int32_t*     mode_ordering; // mode storage ordering
  taco_mode_t* mode_types;    // mode storage types
  uint8_t***   indices;       // tensor index data (per mode)
  uint8_t*     vals;          // tensor values
  int32_t      vals_size;     // values array size
} taco_tensor_t;

static inline
void init_taco_tensor_t(taco_tensor_t* t, int32_t order, int32_t csize,
                        int32_t* dimensions, int32_t* modeOrdering,
                        taco_mode_t* mode_types) {
  t->order         = order;
  t->dimensions    =     (int32_t*)malloc(order * sizeof(int32_t));
  t->mode_ordering =     (int32_t*)malloc(order * sizeof(int32_t));
  t->mode_types    = (taco_mode_t*)malloc(order * sizeof(taco_mode_t));
  t->indices       =   (uint8_t***)malloc(order * sizeof(uint8_t***));
  t->csize         = csize;

  for (int32_t i = 0; i < order; i++) {
    t->dimensions[i]    = dimensions[i];
    t->mode_ordering[i] = modeOrdering[i];
    t->mode_types[i]    = mode_types[i];
    switch (t->mode_types[i]) {
      case taco_mode_dense:
        t->indices[i]    = (uint8_t**)malloc(1 * sizeof(uint8_t**));
      case taco_mode_sparse:
        t->indices[i]    = (uint8_t**)malloc(2 * sizeof(uint8_t**));
    }
  }
}

static inline
void deinit_taco_tensor_t(taco_tensor_t* t) {
  for (int i = 0; i < t->order; i++) {
    free(t->indices[i]);
  }
  free(t->indices);

  free(t->dimensions);
  free(t->mode_ordering);
  free(t->mode_types);
}

#endif
