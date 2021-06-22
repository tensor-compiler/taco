#include "taco_mapper.h"
#include "mappers/logging_wrapper.h"

using namespace Legion;
using namespace Legion::Mapping;

void register_taco_mapper(Machine machine, Runtime *runtime, const std::set<Processor> &local_procs) {
  for (auto it : local_procs) {
#ifdef TACO_USE_LOGGING_MAPPER
    runtime->replace_default_mapper(new Mapping::LoggingWrapper(new TACOMapper(runtime->get_mapper_runtime(), machine, it)), it);
#else
    runtime->replace_default_mapper(new TACOMapper(runtime->get_mapper_runtime(), machine, it), it);
#endif
  }
}

TACOMapper::TACOMapper(Legion::Mapping::MapperRuntime *rt, Legion::Machine &machine, const Legion::Processor &local)
    : DefaultMapper(rt, machine, local) {
  {
    int argc = Legion::HighLevelRuntime::get_input_args().argc;
    char **argv = Legion::HighLevelRuntime::get_input_args().argv;
    for (int i = 1; i < argc; i++) {
#define BOOL_ARG(argname, varname) do {       \
          if (!strcmp(argv[i], (argname))) {    \
            varname = true;                   \
            continue;                         \
          } } while(0);
      BOOL_ARG("-tm:fill_cpu", this->preferCPUFill);
      BOOL_ARG("-tm:validate_cpu", this->preferCPUValidate);
      BOOL_ARG("-tm:disable_mapping_cache", this->disableMappingCache);
      BOOL_ARG("-tm:untrack_valid_regions", this->untrackValidRegions);
#undef BOOL_ARG
    }
  }
}

void TACOMapper::select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                                         const Legion::Task &task,
                                         const SelectShardingFunctorInput &input,
                                         SelectShardingFunctorOutput &output) {
  // See if there is something special that we need to do. Otherwise, return
  // the TACO sharding functor.
  if ((task.tag & PLACEMENT_SHARD) != 0) {
    int *args = (int *) (task.args);
    // TODO (rohany): This logic makes it look like an argument
    //  serializer / deserializer like is done in Legate would be helpful.
    // The shard ID is the first argument. The generated code registers the desired
    // sharding functor before launching the task.
    Legion::ShardingID shardingID = args[0];
    output.chosen_functor = shardingID;
  } else {
    output.chosen_functor = TACOShardingFunctorID;
  }
}

void TACOMapper::map_task(const Legion::Mapping::MapperContext ctx,
                          const Legion::Task &task,
                          const MapTaskInput &input,
                          MapTaskOutput &output) {
  DefaultMapper::map_task(ctx, task, input, output);
  // If the tag is marked for untracked valid regions, then mark all of its
  // read only regions as up for collection.
  if ((task.tag & UNTRACK_VALID_REGIONS) != 0 && this->untrackValidRegions) {
    for (size_t i = 0; i < task.regions.size(); i++) {
      auto &rg = task.regions[i];
      if (rg.privilege == READ_ONLY) {
        output.untracked_valid_regions.insert(i);
      }
    }
  }
}

void TACOMapper::default_policy_select_constraints(Legion::Mapping::MapperContext ctx,
                                                   Legion::LayoutConstraintSet &constraints,
                                                   Legion::Memory target_memory,
                                                   const Legion::RegionRequirement &req) {
  // Ensure that regions are mapped in row-major order.
  Legion::IndexSpace is = req.region.get_index_space();
  Legion::Domain domain = runtime->get_index_space_domain(ctx, is);
  int dim = domain.get_dim();
  std::vector<Legion::DimensionKind> dimension_ordering(dim + 1);
  for (int i = 0; i < dim; ++i) {
    dimension_ordering[dim - i - 1] =
        static_cast<Legion::DimensionKind>(static_cast<int>(LEGION_DIM_X) + i);
  }
  dimension_ordering[dim] = LEGION_DIM_F;
  constraints.add_constraint(Legion::OrderingConstraint(dimension_ordering, false/*contiguous*/));
  DefaultMapper::default_policy_select_constraints(ctx, constraints, target_memory, req);
}

DefaultMapper::CachedMappingPolicy
TACOMapper::default_policy_select_task_cache_policy(Legion::Mapping::MapperContext ctx,
                                                    const Legion::Task &task) {
  if (this->disableMappingCache) {
    return DEFAULT_CACHE_POLICY_DISABLE;
  }
  return DEFAULT_CACHE_POLICY_ENABLE;
}

void TACOMapper::default_policy_select_target_processors(
    Legion::Mapping::MapperContext ctx,
    const Legion::Task &task,
    std::vector<Legion::Processor> &target_procs) {
  // TODO (rohany): Add a TACO tag to the tasks.
  if (task.is_index_space) {
    // Index launches should be placed directly on the processor
    // they were sliced to.
    target_procs.push_back(task.target_proc);
  } else if (std::string(task.get_task_name()).find("task_") != std::string::npos) {
    // Other point tasks should stay on the originating processor, if they are
    // using a CPU Proc. Otherwise, send the tasks where the default mapper
    // says they should go. I think that the heuristics for OMP_PROC and TOC_PROC
    // are correct for our use case.
    if (task.target_proc.kind() == task.orig_proc.kind()) {
      target_procs.push_back(task.orig_proc);
    } else {
      DefaultMapper::default_policy_select_target_processors(ctx, task, target_procs);
    }
  } else {
    DefaultMapper::default_policy_select_target_processors(ctx, task, target_procs);
  }
}

int TACOMapper::default_policy_select_garbage_collection_priority(Legion::Mapping::MapperContext ctx, MappingKind kind,
                                                                  Legion::Memory memory,
                                                                  const Legion::Mapping::PhysicalInstance &instance,
                                                                  bool meets_fill_constraints, bool reduction) {
  // Copy the default mapper's heuristic to eagerly collection reduction instances.
  if (reduction) {
    return LEGION_GC_FIRST_PRIORITY;
  }
  // Deviate from the default mapper to give all instances default GC priority. The
  // default mapper most of the time marks instances as un-collectable from the GC,
  // which leads to problems when using instances in a "temporary buffer" style.
  return LEGION_GC_DEFAULT_PRIORITY;
}

std::vector<Legion::Processor> TACOMapper::select_targets_for_task(const Legion::Mapping::MapperContext ctx,
                                                       const Legion::Task& task) {
  auto kind = this->default_find_preferred_variant(task, ctx, false /* needs tight bounds */).proc_kind;
  // If we're requested to fill/validate on the CPU, then hijack the initial
  // processor selection to do so.
  if ((this->preferCPUFill && task.task_id == TID_TACO_FILL_TASK) ||
      (this->preferCPUValidate && task.task_id == TID_TACO_VALIDATE_TASK)) {
    // See if we have any OMP procs.
    auto targetKind = Legion::Processor::Kind::LOC_PROC;
    Legion::Machine::ProcessorQuery omps(this->machine);
    omps.only_kind(Legion::Processor::OMP_PROC);
    if (omps.count() > 0) {
      targetKind = Legion::Processor::Kind::OMP_PROC;
    }
    kind = targetKind;
  }
  // We always map to the same address space if replication is enabled.
  auto sameAddressSpace = ((task.tag & DefaultMapper::SAME_ADDRESS_SPACE) != 0) || this->replication_enabled;
  if (sameAddressSpace) {
    // If we are meant to stay local, then switch to return the appropriate
    // cached processors.
    switch (kind) {
      case Legion::Processor::OMP_PROC: {
        return this->local_omps;
      }
      case Legion::Processor::TOC_PROC: {
        return this->local_gpus;
      }
      case Legion::Processor::LOC_PROC: {
        return this->local_cpus;
      }
      default: {
        assert(false);
      }
    }
  } else {
    // If we are meant to distribute over all of the processors, then run a query
    // to find all processors of the desired kind.
    Legion::Machine::ProcessorQuery all_procs(machine);
    all_procs.only_kind(kind);
    return std::vector<Legion::Processor>(all_procs.begin(), all_procs.end());
  }

  // Keep the compiler happy.
  assert(false);
  return {};
}

void TACOMapper::slice_task(const Legion::Mapping::MapperContext ctx,
                            const Legion::Task &task,
                            const SliceTaskInput &input,
                            SliceTaskOutput &output) {
  if (task.tag & PLACEMENT) {
    // Placement tasks will put the dimensions of the placement grid at the beginning
    // of the task arguments. Here, we extract the packed placement grid dimensions.
    int dim = input.domain.get_dim();
    int *args = (int *) (task.args);
    std::vector<int> gridDims(dim);
    for (int i = 0; i < dim; i++) {
      gridDims[i] = args[i];
    }
    auto targets = this->select_targets_for_task(ctx, task);
    switch (dim) {
#define BLOCK(DIM) \
        case DIM:  \
          {        \
            Legion::DomainT<DIM, Legion::coord_t> pointSpace = input.domain; \
            this->decompose_points(pointSpace, gridDims, targets, output.slices);        \
            break;   \
          }
      LEGION_FOREACH_N(BLOCK)
#undef BLOCK
      default:
        assert(false);
    }
  } else {
    // Otherwise, we have our own implementation of slice task. The reason for this is
    // because the default mapper gets confused and hits a cache of domain slices. This
    // messes up the placement that we are going for with the index launches. This
    // implementation mirrors the standard slicing strategy of the default mapper.
    auto targets = this->select_targets_for_task(ctx, task);
    switch (input.domain.get_dim()) {
#define BLOCK(DIM) \
        case DIM:  \
          {        \
            Legion::DomainT<DIM,Legion::coord_t> point_space = input.domain; \
            Legion::Point<DIM,Legion::coord_t> num_blocks = \
              default_select_num_blocks<DIM>(targets.size(), point_space.bounds); \
            this->default_decompose_points<DIM>(point_space, targets, \
                  num_blocks, false/*recurse*/, \
                  stealing_enabled, output.slices); \
            break;   \
          }
      LEGION_FOREACH_N(BLOCK)
#undef BLOCK
      default:
        assert(false);
    }
  }
}
