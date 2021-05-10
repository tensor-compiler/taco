#include "taco_mapper.h"

using namespace Legion;

void register_taco_mapper(Machine machine, Runtime *runtime, const std::set<Processor> &local_procs) {
  for (auto it : local_procs) {
    runtime->replace_default_mapper(new TACOMapper(runtime->get_mapper_runtime(), machine, it), it);
  }
}