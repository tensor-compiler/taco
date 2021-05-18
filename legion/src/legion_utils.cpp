#include <functional>
#include <chrono>
#include <iostream>

#include "legion.h"
#include "legion_utils.h"

using namespace Legion;

Legion::PhysicalRegion getRegionToWrite(Legion::Context ctx, Legion::Runtime* runtime, Legion::LogicalRegion r, Legion::LogicalRegion parent) {
  Legion::RegionRequirement req(r, READ_WRITE, EXCLUSIVE, parent);
  req.add_field(FID_VAL);
  return runtime->map_region(ctx, req);
}

void benchmark(std::function<void(void)> f) {
  auto start = std::chrono::high_resolution_clock::now();
  f();
  auto end = std::chrono::high_resolution_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "Execution time: " << ms << " ms." << std::endl;
}

#ifndef TACO_USE_CUDA
// Dummy implementation of initCuBLAS if we aren't supposed to use CUDA.
void initCuBLAS(Context ctx, Runtime* runtime) {}
#endif
