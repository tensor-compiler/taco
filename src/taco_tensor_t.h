/// This file defines the runtime struct used to pass raw tensors to generated
/// code.  Note: this file must be valid C99, not C++.

#ifndef TACO_TENSOR_T_DEFINED
#define TACO_TENSOR_T_DEFINED
typedef enum { taco_level_dense, taco_level_sparse } taco_level_t;

typedef struct {
  int order;                 // order of the tensor (i.e. how many dimensions)
  
  int* levels;               // the ordering of the levels
  taco_level_t* levelTypes;  // for each level, the type of level it is (Dense, Sparse, etc)
  int* levelSize;            // for each level, the size of that level
                             // (the logical size of the corresponding dimension)
  
  int** pos;                // an array of pointers, each pointer points to a level's "pos" array
  int** idxs;                // an array of pointers, each pointer points to a level's "indices" array
  
  int elem_size;             // the size of an element, in bytes.  currently only 8 is supported (doubles)
  uint8_t* vals;             // a pointer to the values array

} taco_tensor_t;
