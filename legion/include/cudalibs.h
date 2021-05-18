#ifndef TACO_CUDALIBS_H
#define TACO_CUDALIBS_H

#include "legion.h"
#include "cublas_v2.h"

// Macro for easily checking the result of CuBLAS calls.
#define CHECK_CUBLAS(expr)                    \
  {                                           \
    cublasStatus_t result = (expr);           \
    checkCuBLAS(result, __FILE__, __LINE__); \
  }

// CuBLAS error checker.
void checkCuBLAS(cublasStatus_t status, const char* file, int line);

// Get and potentially initialize CuBLAS handle for this processor.
cublasHandle_t getCuBLAS();
void initCUDA();
#endif // TACO_CUDALIBS_H
