#include "taco_mapper.h"
#include "mappers/logging_wrapper.h"

using namespace Legion;

void register_taco_mapper(Machine machine, Runtime *runtime, const std::set<Processor> &local_procs) {
  for (auto it : local_procs) {
#ifdef TACO_USE_LOGGING_MAPPER
    runtime->replace_default_mapper(new Mapping::LoggingWrapper(new TACOMapper(runtime->get_mapper_runtime(), machine, it)), it);
#else
    runtime->replace_default_mapper(new TACOMapper(runtime->get_mapper_runtime(), machine, it), it);
#endif
  }
}