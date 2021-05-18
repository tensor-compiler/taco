#include "fill.cuh"

// Register the fill task for several common types that we use.
template void registerGPUFillTask<int32_t>();
template void registerGPUFillTask<int64_t>();
template void registerGPUFillTask<float>();
template void registerGPUFillTask<double>();
