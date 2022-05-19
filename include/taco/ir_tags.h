#ifndef TACO_IR_TAGS_H
#define TACO_IR_TAGS_H

namespace taco {

/// ParallelUnit::CPUThread generates a pragma to parallelize over CPU threads
/// ParallelUnit::CPUVector generates a pragma to utilize a CPU vector unit
/// ParallelUnit::GPUBlock must be used with GPUThread to create blocks of GPU threads
/// ParallelUnit::GPUWarp can be optionally used to allow for GPU warp-level primitives
/// ParallelUnit::GPUThread causes for every iteration to be executed on a separate GPU thread
enum class ParallelUnit {
  NotParallel, DefaultUnit, GPUBlock, GPUWarp, GPUThread, CPUThread, CPUVector, CPUThreadGroupReduction, GPUBlockReduction, GPUWarpReduction
};
extern const char *ParallelUnit_NAMES[];

/// OutputRaceStrategy::NoRaces raises a compile-time error if an output race exists
/// OutputRaceStrategy::Atomics replace racing instructions with atomics
/// OutputRaceStrategy::Temporary uses a temporary array for outputs that is serially reduced
/// OutputRaceStrategy::ParallelReduction uses reduction operations across a warp/vector
/// OutputRaceStrategy::IgnoreRaces allows the user to specify that races can be safely ignored
enum class OutputRaceStrategy {
  IgnoreRaces, NoRaces, Atomics, Temporary, ParallelReduction
};
extern const char *OutputRaceStrategy_NAMES[];

enum class BoundType {
  MinExact, MinConstraint, MaxExact, MaxConstraint
};
extern const char *BoundType_NAMES[];

enum class AssembleStrategy {
  Append, Insert
};
extern const char *AssembleStrategy_NAMES[];

/// MergeStrategy::TwoFinger merges iterators by incrementing one at a time
/// MergeStrategy::Galloping merges iterators by exponential search (galloping)
enum class MergeStrategy {
  TwoFinger, Gallop
};
extern const char *MergeStrategy_NAMES[];

}

#endif //TACO_IR_TAGS_H
