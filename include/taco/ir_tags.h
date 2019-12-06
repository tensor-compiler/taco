#ifndef TACO_IR_TAGS_H
#define TACO_IR_TAGS_H

namespace taco {
enum class ParallelUnit {
  NotParallel, DefaultUnit, GPUBlock, GPUWarp, GPUThread, CPUThread, CPUVector, CPUThreadGroupReduction, GPUBlockReduction, GPUWarpReduction
};
extern const char *ParallelUnit_NAMES[];

enum class OutputRaceStrategy {
  IgnoreRaces, NoRaces, Atomics, Temporary, ParallelReduction
};
extern const char *OutputRaceStrategy_NAMES[];

enum class BoundType {
  MinExact, MinConstraint, MaxExact, MaxConstraint
};
extern const char *BoundType_NAMES[];
}

#endif //TACO_IR_TAGS_H
