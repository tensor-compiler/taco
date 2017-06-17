/// This file defines the runtime struct used to pass raw tensors to generated
/// code.  Note: this file must be valid C99, not C++.
/// This *must* be kept in sync with the version used in codegen_c.cpp

#ifndef TACO_TENSOR_T_DEFINED
#define TACO_TENSOR_T_DEFINED

typedef enum { taco_dim_dense, taco_dim_sparse } taco_dim_t;

typedef struct {
  int32_t     order;      // tensor order (number of dimensions)
  int32_t*    dims;       // tensor dimensions
  taco_dim_t* dim_types;  // dimension storage types
  int32_t     csize;      // component size
  
  int32_t*    dim_order;  // dimension storage order
  uint8_t***  indices;    // tensor index data (per dimension)
  uint8_t*    vals;       // tensor values
} taco_tensor_t;

#endif
