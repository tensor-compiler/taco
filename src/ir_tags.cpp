#include "taco/ir_tags.h"

namespace taco {
const char *PARALLEL_UNIT_NAMES[] = {"NOT_PARALLEL", "DEFAULT_UNIT", "GPU_BLOCK", "GPU_WARP", "GPU_THREAD", "CPU_THREAD", "CPU_VECTOR"};
const char *OUTPUT_RACE_STRATEGY_NAMES[] = {"IGNORE_RACES", "NO_RACES", "ATOMICS", "REDUCTION"};
}