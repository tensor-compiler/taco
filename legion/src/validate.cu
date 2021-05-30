#include "validate.cuh"

// Register the validate task for several common types that we use.
template void registerGPUValidateTask<int32_t>();
template void registerGPUValidateTask<int64_t>();
template void registerGPUValidateTask<float>();
template void registerGPUValidateTask<double>();
