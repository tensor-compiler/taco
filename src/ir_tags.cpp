#include "taco/ir_tags.h"

namespace taco {

const char *ParallelUnit_NAMES[] = {"NotParallel", "DefaultUnit", "GPUBlock", "GPUWarp", "GPUThread", "CPUThread", "CPUVector", "CPUThreadGroupReduction", "GPUBlockReduction", "GPUWarpReduction", "Distributed", "DistributedGPU", "LegionReduction"};
const char *OutputRaceStrategy_NAMES[] = {"IgnoreRaces", "NoRaces", "Atomics", "Temporary", "ParallelReduction"};
const char *BoundType_NAMES[] = {"MinExact", "MinConstraint", "MaxExact", "MaxConstraint"};
const char *AssembleStrategy_NAMES[] = {"Append", "Insert"};

bool distributedParallelUnit(ParallelUnit unit) {
  return unit == ParallelUnit::DistributedGPU || unit == ParallelUnit ::DistributedNode;
}

}
