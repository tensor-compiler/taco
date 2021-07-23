#include <functional>
#include <chrono>
#include <iostream>
#include "stdio.h"

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

void benchmark(Legion::Context ctx, Legion::Runtime* runtime, std::function<void(void)> f) {
  auto start = std::chrono::high_resolution_clock::now();
  f();
  auto end = std::chrono::high_resolution_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  LEGION_PRINT_ONCE(runtime, ctx, stdout, "Execution time: %lld ms.\n", ms);
}

void benchmark(Legion::Context ctx, Legion::Runtime* runtime, std::vector<size_t>& times, std::function<void(void)> f) {
  auto start = std::chrono::high_resolution_clock::now();
  f();
  auto end = std::chrono::high_resolution_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  LEGION_PRINT_ONCE(runtime, ctx, stdout, "Execution time: %lld ms.\n", ms);
  times.push_back(ms);
}

void benchmarkAsyncCall(Legion::Context ctx, Legion::Runtime* runtime, std::vector<size_t>& times, std::function<void(void)> f) {
  auto start = runtime->get_current_time(ctx);
  f();
  runtime->issue_execution_fence(ctx);
  auto end = runtime->get_current_time(ctx);
  auto ms = size_t((end.get<double>() - start.get<double>()) * 1e3);
  LEGION_PRINT_ONCE(runtime, ctx, stdout, "Execution time: %ld ms.\n", ms);
  times.push_back(ms);
}

size_t getGEMMFLOPCount(size_t M, size_t N, size_t K) {
  return M * N * (2 * K - 1);
}

size_t getTTMCFLOPCount(size_t I, size_t J, size_t K, size_t L) {
  return I * getGEMMFLOPCount(J, K, L);
}

size_t getMTTKRPFLOPCount(size_t I, size_t J, size_t K, size_t L) {
  return I * getGEMMFLOPCount(J, K, L) + 2 * (I * J * L);
}

double getGFLOPS(size_t flopCount, size_t ms) {
  double s = double(ms) / 1e3;
  double GFLOP = double(flopCount) / 1e9;
  return GFLOP / s;
}

#ifndef TACO_USE_CUDA
// Dummy implementations of initCuBLAS and initCUDA if we aren't supposed to use CUDA.
void initCuBLAS(Context ctx, Runtime* runtime) {}
void initCUDA() {}
#endif
