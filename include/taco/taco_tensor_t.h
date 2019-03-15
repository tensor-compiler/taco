/// This file defines the runtime struct used to pass raw tensors to generated
/// code.  Note: this file must be valid C99, not C++.
/// This *must* be kept in sync with the version used in codegen_c.cpp
/// TODO: Remove `vals_size` after old lowering machinery has been replaced.

#ifndef TACO_TENSOR_T_DEFINED
#define TACO_TENSOR_T_DEFINED

#include <cstdint>

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

taco_tensor_t *init_taco_tensor_t(int32_t order, int32_t csize,
                        int32_t* dimensions, int32_t* modeOrdering,
                        taco_mode_t* mode_types);

void deinit_taco_tensor_t(taco_tensor_t* t);

#endif
