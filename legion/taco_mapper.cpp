#include "taco_mapper.h"

static void register_taco_mapper(Machine machine, Runtime *runtime, const std::set<Processor> &local_procs) {
  for(auto it = local_procs.begin(); it != local_procs.end(); it++) {
    runtime->replace_default_mapper(new TACOMapper(runtime->get_mapper_runtime(), machine, *it), *it);
  }
}