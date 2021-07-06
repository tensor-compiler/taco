#include "taco/ir_tags.h"

namespace taco {

const char *ParallelUnit_NAMES[] = {"NotParallel", "DefaultUnit", "GPUBlock", "GPUWarp", "GPUThread", "CPUThread",
                                    "CPUVector", "CPUThreadGroupReduction", "GPUBlockReduction", "GPUWarpReduction", "Spatial"};
const char *OutputRaceStrategy_NAMES[] = {"IgnoreRaces", "NoRaces", "Atomics", "Temporary", "ParallelReduction",
                                          "SpatialReduction"};
const char *BoundType_NAMES[] = {"MinExact", "MinConstraint", "MaxExact", "MaxConstraint"};
const char *AssembleStrategy_NAMES[] = {"Append", "Insert"};
const char *MemoryLocation_NAMES[] = {"Default", "GPUSharedMemory", "SpatialDRAM", "SpatialSparseDRAM",
                                      "SpatialSRAM", "SpatialSparseSRAM", "SpatialSparseParSRAM", "SpatialFIFO",
                                      "SpatialReg", "SpatialArgIn", "SpatialArgOut"};

// Spatial only
const char *SpatialRMWOperators_NAMES[] = {"Read", "Write", "Add", "Swap"};
const char *SpatialRMWOperators_IR[] = {"read", "write", "add", "swap"};

const char *SpatialMemOrdering_NAMES[] = {"Unordered", "Ordered"};
const char *SpatialMemOrdering_IR[] = {"unordered", "ordered"};

}
