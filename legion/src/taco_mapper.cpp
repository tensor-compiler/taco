#include "taco_mapper.h"
#include "mappers/logging_wrapper.h"

using namespace Legion;
using namespace Legion::Mapping;

const char* TACOMapperName = "TACOMapper";

void register_taco_mapper(Machine machine, Runtime *runtime, const std::set<Processor> &local_procs) {
  // If we're supposed to backpressure task executions, then we need to only
  // have a single mapper per node. Otherwise, we can use a mapper per processor.
  bool backpressure = false;
  auto args = Legion::Runtime::get_input_args();
  for (int i = 1; i < args.argc; i++) {
    if (strcmp(args.argv[i], "-tm:enable_backpressure") == 0) {
      backpressure = true;
      break;
    }
  }

  if (backpressure) {
    auto proc = *local_procs.begin();
#ifdef TACO_USE_LOGGING_MAPPER
    runtime->replace_default_mapper(new Mapping::LoggingWrapper(new TACOMapper(runtime->get_mapper_runtime(), machine, proc, TACOMapperName)), Processor::NO_PROC);
#else
    runtime->replace_default_mapper(new TACOMapper(runtime->get_mapper_runtime(), machine, proc, TACOMapperName), Processor::NO_PROC);
#endif
  } else {
    for (auto it : local_procs) {
#ifdef TACO_USE_LOGGING_MAPPER
      runtime->replace_default_mapper(new Mapping::LoggingWrapper(new TACOMapper(runtime->get_mapper_runtime(), machine, it, TACOMapperName)), it);
#else
      runtime->replace_default_mapper(new TACOMapper(runtime->get_mapper_runtime(), machine, it, TACOMapperName), it);
#endif
    }
  }
}

TACOMapper::TACOMapper(Legion::Mapping::MapperRuntime *rt, Legion::Machine &machine, const Legion::Processor &local, const char* name)
    : DefaultMapper(rt, machine, local, name) {
  {
    int argc = Legion::HighLevelRuntime::get_input_args().argc;
    char **argv = Legion::HighLevelRuntime::get_input_args().argv;
    for (int i = 1; i < argc; i++) {
#define BOOL_ARG(argname, varname) do {       \
          if (!strcmp(argv[i], (argname))) {  \
            varname = true;                   \
            continue;                         \
          } } while(0);
#define INT_ARG(argname, varname) do {      \
          if (!strcmp(argv[i], (argname))) {  \
            varname = atoi(argv[++i]);      \
            continue;                       \
          } } while(0);
      BOOL_ARG("-tm:fill_cpu", this->preferCPUFill);
      BOOL_ARG("-tm:validate_cpu", this->preferCPUValidate);
      BOOL_ARG("-tm:untrack_valid_regions", this->untrackValidRegions);
      BOOL_ARG("-tm:numa_aware_alloc", this->numaAwareAllocs);
      BOOL_ARG("-tm:enable_backpressure", this->enableBackpressure);
      INT_ARG("-tm:backpressure_max_in_flight", this->maxInFlightTasks);
      BOOL_ARG("-tm:multiple_shards_per_node", this->multipleShardsPerNode);
#undef BOOL_ARG
    }
  }

  // Record for each OpenMP processor what NUMA region is the closest.
  for (auto proc : this->local_omps) {
    Machine::MemoryQuery local(this->machine);
    local.local_address_space()
         .only_kind(Memory::SOCKET_MEM)
         .best_affinity_to(proc)
         ;
    if (local.count() > 0) {
      this->numaDomains[proc] = local.first();
    }
  }

  if (this->multipleShardsPerNode) {
    // If we have GPUs, map each local CPU to a local GPU corresponding to each shard.
    // If we actually have GPUs, then the number of GPUs should be equal to the number of CPUs.
    // Note that this makes the most sense to do in map_replicate_task, but that is only executed
    // on a single mapper! Therefore, we need to do this in initialization so that each mapper has
    // this data structure populated.
    if (this->local_gpus.size() > 0) {
      assert(this->local_cpus.size() == this->local_gpus.size());
      auto cpu = this->local_cpus.begin();
      auto gpu = this->local_gpus.begin();
      // Construct a mapping between each shard and a GPU.
      for (; cpu != this->local_cpus.end() && gpu != this->local_gpus.end(); cpu++,gpu++) {
        this->shardCPUGPUMapping[*cpu] = *gpu;
      }
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

void TACOMapper::select_task_options(const Legion::Mapping::MapperContext ctx,
                                     const Legion::Task &task,
                                     TaskOptions &output) {
  DefaultMapper::select_task_options(ctx, task, output);
  // Override the default options if we are supposed to run multiple
  // shards per node.
  if (this->multipleShardsPerNode && task.get_depth() == 0) {
    output.replicate = true;
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
  // Mark that we want profiling from this task if we're supposed to backpressure it.
  if ((task.tag & BACKPRESSURE_TASK) != 0 && this->enableBackpressure) {
    output.task_prof_requests.add_measurement<ProfilingMeasurements::OperationStatus>();
  }
}

// map_replicate_task is overridden for situations where we are running
// multiple target processors per node and need to arrange them into
// the multi-dimensional grids of our choice. In these cases, we want to
// use multiple shards per node (i.e. 1 shard per target processor).
void TACOMapper::map_replicate_task(const Legion::Mapping::MapperContext ctx, const Legion::Task &task,
                                    const MapTaskInput &input, const MapTaskOutput &default_output,
                                    MapReplicateTaskOutput &output) {
  // If we aren't expected to run multiple shards per node, then just
  // fall back to the default mapper.
  if (!this->multipleShardsPerNode) {
    DefaultMapper::map_replicate_task(ctx, task, input, default_output, output);
    return;
  }

  // We should only be mapping the top level task.
  assert((task.get_depth() == 0) && (task.regions.size() == 0));
  assert(task.target_proc.kind() == Processor::LOC_PROC);
  auto targetKind = Processor::LOC_PROC;
  const auto chosen = default_find_preferred_variant(task, ctx, true /* needs tight bound */, true /* cache */, targetKind);
  assert(chosen.is_replicable);
  // Collect all LOC_PROC's to put shards on.
  Legion::Machine::ProcessorQuery cpuQuery(this->machine);
  cpuQuery.only_kind(targetKind);
  auto allCPUs = std::vector<Processor>(cpuQuery.begin(), cpuQuery.end());
  // We also need all of the GPUs so that we can map each shard to its GPU.
  // const auto remoteGPUs = remote_procs_by_kind(Processor::TOC_PROC);

  // Create a shard for each CPU processor.
  output.task_mappings.resize(allCPUs.size());
  output.control_replication_map.resize(allCPUs.size());
  for (size_t i = 0; i < allCPUs.size(); i++) {
    output.task_mappings[i].target_procs.push_back(allCPUs[i]);
    output.task_mappings[i].chosen_variant = chosen.variant;
    output.control_replication_map[i] = allCPUs[i];
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

Memory TACOMapper::default_policy_select_target_memory(Legion::Mapping::MapperContext ctx,
                                                       Legion::Processor target_proc,
                                                       const Legion::RegionRequirement &req,
                                                       Legion::MemoryConstraint mc) {
  // If we are supposed to perform NUMA aware allocations
  if (target_proc.kind() == Processor::OMP_PROC && this->numaAwareAllocs) {
    auto it = this->numaDomains.find(target_proc);
    assert(it != this->numaDomains.end());
    return it->second;
  } else {
    return DefaultMapper::default_policy_select_target_memory(ctx, target_proc, req, mc);
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

  // If we're running with multiple shards per node, then we already have a decomposition of tasks
  // onto each shard. So, just return the assigned processor. Note that we only do this for tasks
  // that are being sharded -- i.e. have depth 1 and are index space launches.
  if (this->multipleShardsPerNode && !this->shardCPUGPUMapping.empty() && task.get_depth() == 1 && task.is_index_space) {
    auto targetProc = this->shardCPUGPUMapping[task.orig_proc];
    assert(kind == targetProc.kind());
    return std::vector<Processor>{targetProc};
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

void TACOMapper::report_profiling(const MapperContext ctx,
                                  const Task& task,
                                  const TaskProfilingInfo& input) {
  // We should only get profiling responses if we've enabled backpressuring.
  assert(this->enableBackpressure);
  // We should only get profiling responses for tasks that are supposed to be backpressured.
  assert((task.tag & BACKPRESSURE_TASK) != 0);
  auto prof = input.profiling_responses.get_measurement<ProfilingMeasurements::OperationStatus>();
  // All our tasks should complete successfully.
  assert(prof->result == Realm::ProfilingMeasurements::OperationStatus::COMPLETED_SUCCESSFULLY);
  // Clean up after ourselves.
  delete prof;
  // Backpressured tasks are launched in a loop, and are kept on the originating processor.
  // So, we'll use orig_proc to index into the queue.
  auto& inflight = this->backPressureQueue[task.orig_proc];
  MapperEvent event;
  // Find this task in the queue.
  for (auto it = inflight.begin(); it != inflight.end(); it++) {
    if (it->id == task.get_unique_id()) {
      event = it->event;
      inflight.erase(it);
      break;
    }
  }
  // Assert that we found a valid event.
  assert(event.exists());
  // Finally, trigger the event for anyone waiting on it.
  this->runtime->trigger_mapper_event(ctx, event);
}

// In select_tasks_to_map, we attempt to perform backpressuring on tasks that
// need to be backpressured.
void TACOMapper::select_tasks_to_map(const MapperContext ctx,
                                     const SelectMappingInput& input,
                                           SelectMappingOutput& output) {
  if (!this->enableBackpressure) {
    DefaultMapper::select_tasks_to_map(ctx, input, output);
  } else {
    // Mark when we are potentially scheduling tasks.
    auto schedTime = std::chrono::high_resolution_clock::now();
    // Create an event that we will return in case we schedule nothing.
    MapperEvent returnEvent;
    // Also maintain a time point of the best return event. We want this function
    // to get invoked as soon as any backpressure task finishes, so we'll use the
    // completion event for the earliest one.
    auto returnTime = std::chrono::high_resolution_clock::time_point::max();

    // Find the depth of the deepest task.
    int max_depth = 0;
    for (std::list<const Task*>::const_iterator it =
        input.ready_tasks.begin(); it != input.ready_tasks.end(); it++)
    {
      int depth = (*it)->get_depth();
      if (depth > max_depth)
        max_depth = depth;
    }
    unsigned count = 0;
    // Only schedule tasks from the max depth in any pass.
    for (std::list<const Task*>::const_iterator it =
        input.ready_tasks.begin(); (count < max_schedule_count) &&
                                   (it != input.ready_tasks.end()); it++)
    {
      auto task = *it;
      bool schedule = true;
      if ((task->tag & BACKPRESSURE_TASK) != 0) {
        // See how many tasks we have in flight. Again, we use the orig_proc here
        // rather than target_proc to match with our heuristics for where serial task
        // launch loops go.
        auto inflight = this->backPressureQueue[task->orig_proc];
        if (inflight.size() == this->maxInFlightTasks) {
          // We've hit the cap, so we can't schedule any more tasks.
          schedule = false;
          // As a heuristic, we'll wait on the first mapper event to
          // finish, as it's likely that one will finish first. We'll also
          // try to get a task that will complete before the current best.
          auto front = inflight.front();
          if (front.schedTime < returnTime) {
            returnEvent = front.event;
            returnTime = front.schedTime;
          }
        } else {
          // Otherwise, we can schedule the task. Create a new event
          // and queue it up on the processor.
          this->backPressureQueue[task->orig_proc].push_back({
            .id = task->get_unique_id(),
            .event = this->runtime->create_mapper_event(ctx),
            .schedTime = schedTime,
          });
        }
      }
      // Schedule tasks that are valid and have the target depth.
      if (schedule && (*it)->get_depth() == max_depth)
      {
        output.map_tasks.insert(*it);
        count++;
      }
    }
    // If we didn't schedule any tasks, tell the runtime to ask us again when
    // our return event triggers.
    if (output.map_tasks.empty()) {
      assert(returnEvent.exists());
      output.deferral_event = returnEvent;
    }
  }
}

Mapper::MapperSyncModel TACOMapper::get_mapper_sync_model() const {
  // If we're going to attempt to backpressure tasks, then we need to use
  // a sync model with high gaurantees.
  if (this->enableBackpressure) {
    return SERIALIZED_NON_REENTRANT_MAPPER_MODEL;
  }
  // Otherwise, we can do whatever the default mapper is doing.
  return DefaultMapper::get_mapper_sync_model();
}
