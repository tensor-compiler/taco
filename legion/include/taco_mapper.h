#ifndef TACO_MAPPER_H
#define TACO_MAPPER_H

#include "legion.h"
#include "mappers/default_mapper.h"

#include "task_ids.h"
#include "shard.h"

#include <chrono>

class TACOMapper : public Legion::Mapping::DefaultMapper {
public:
  // Mapping tags handled specific for the TACO mapper.
  enum MappingTags {
    // Indicates that this task launch is used for data placement.
    PLACEMENT = (1 << 5),
    // Indicates that this task launch is used for data placement, but the placement
    // will be handled by sharding functors rather than slice_task.
    PLACEMENT_SHARD = (1 << 6),
    // Marks that the task should have its read-only regions be eligible for collection.
    UNTRACK_VALID_REGIONS = (1 << 7),
    // Marks that invocations of this task should be backpressured by the mapper.
    BACKPRESSURE_TASK = (1 << 8),
  };

  TACOMapper(Legion::Mapping::MapperRuntime *rt, Legion::Machine &machine, const Legion::Processor &local, const char* name);

  void select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                               const Legion::Task &task,
                               const SelectShardingFunctorInput &input,
                               SelectShardingFunctorOutput &output) override;

  void map_task(const Legion::Mapping::MapperContext ctx,
                const Legion::Task &task,
                const MapTaskInput &input,
                MapTaskOutput &output) override;

  void default_policy_select_constraints(Legion::Mapping::MapperContext ctx,
                                         Legion::LayoutConstraintSet &constraints,
                                         Legion::Memory target_memory,
                                         const Legion::RegionRequirement &req) override;

  void default_policy_select_target_processors(
      Legion::Mapping::MapperContext ctx,
      const Legion::Task &task,
      std::vector<Legion::Processor> &target_procs) override;

  int default_policy_select_garbage_collection_priority(
      Legion::Mapping::MapperContext ctx,
      MappingKind kind, Legion::Memory memory,
      const Legion::Mapping::PhysicalInstance &instance,
      bool meets_fill_constraints, bool reduction) override;

  void slice_task(const Legion::Mapping::MapperContext ctx,
                  const Legion::Task &task,
                  const SliceTaskInput &input,
                  SliceTaskOutput &output) override;

  Legion::Memory default_policy_select_target_memory(Legion::Mapping::MapperContext ctx,
                                                     Legion::Processor target_proc,
                                                     const Legion::RegionRequirement &req,
                                                     Legion::MemoryConstraint mc = Legion::MemoryConstraint()) override;

  void report_profiling(const Legion::Mapping::MapperContext ctx,
                        const Legion::Task& task,
                        const TaskProfilingInfo& input) override;

  void select_tasks_to_map(const Legion::Mapping::MapperContext ctx,
                           const SelectMappingInput& input,
                                 SelectMappingOutput& output) override;

  MapperSyncModel get_mapper_sync_model() const override;

  template<int DIM>
  void decompose_points(const Legion::DomainT<DIM, Legion::coord_t> &point_space,
                        std::vector<int>& gridDims,
			std::vector<Legion::Processor> targets,
                        std::vector<TaskSlice> &slices) {
    slices.reserve(point_space.volume());

    // We'll allocate each node a point in the index space.
    auto node = 0;

    // We'll iterate over the full placement grid.
    Legion::Rect<DIM, Legion::coord_t> procRect;
    for (int i = 0; i < DIM; i++) {
      procRect.lo[i] = 0;
      procRect.hi[i] = gridDims[i] - 1;
    }

    for (Legion::PointInRectIterator<DIM> itr(procRect); itr(); itr++) {
      // Always increment the node counter -- we want to possibly skip nodes
      // when we have a Face() placement restriction.
      auto curNode = node++;
      auto point = *itr;
      // We'll skip any points that don't align with the faces.
      if (!point_space.contains(point)) {
        continue;
      }

      // Construct the output slice for Legion.
      Legion::DomainT<DIM, Legion::coord_t> slice;
      slice.bounds.lo = point;
      slice.bounds.hi = point;
      slice.sparsity = point_space.sparsity;
      if (!slice.dense()) { slice = slice.tighten(); }
      if (slice.volume() > 0) {
        TaskSlice ts;
        ts.domain = slice;
        ts.proc = targets[curNode % targets.size()];
        ts.recurse = false;
        ts.stealable = false;
        slices.push_back(ts);
      }
    }
  }

private:
  // Helper method to choose the what processors to slice a task launch onto.
  std::vector<Legion::Processor> select_targets_for_task(const Legion::Mapping::MapperContext ctx,
                                                         const Legion::Task &task);

  // Denotes whether the fill operation should place data onto CPU memories
  // or GPU memories.
  bool preferCPUFill = false;
  // Same as preferCPUFill but for validation.
  bool preferCPUValidate = false;
  // Denotes whether read-only valid regions of leaf tasks should be marked
  // eagerly for collection.
  bool untrackValidRegions = false;
  // Denotes whether to perform region allocations in each numa node.
  bool numaAwareAllocs = false;
  // A map from OpenMP processor to the NUMA region local to that OpenMP processor.
  std::map<Legion::Processor, Legion::Memory> numaDomains;

  // Denotes whether or not the mapper should attempt to backpressure executions of
  // tagged tasks.
  bool enableBackpressure = false;
  // Denotes how many tasks being backpressured can execute at the same time.
  size_t maxInFlightTasks = 1;

  // InFlightTask represents a task currently being executed.
  struct InFlightTask {
    // Unique identifier of the task instance.
    Legion::UniqueID id;
    // An event that will be triggered when the task finishes.
    Legion::Mapping::MapperEvent event;
    // A clock measurement from when the task was scheduled.
    std::chrono::high_resolution_clock::time_point schedTime;
  };
  // backPressureQueue maintains state for each processor about how many
  // tasks that are marked to be backpressured are executing on the processor.
  // TODO (rohany): This works in a situation where we're running one of the
  //  benchmark applications, and only single task at a time is being backpressured/
  //  serialized onto a processor. If there are multiple different kernels running
  //  that all need to be backpressured then I'm not sure how that will work.
  std::map<Legion::Processor, std::deque<InFlightTask>> backPressureQueue;

  // TODO (rohany): It may end up being necessary that we need to explicitly map
  //  regions for placement tasks. If so, Manolis says the following approach
  //  is the right thing:
  //  * Slice tasks sends tasks where they are supposed to go.
  //  * Map task needs to create a new instance of the region visible to the
  //    target processor.
};

// Register the TACO mapper.
void register_taco_mapper(Legion::Machine machine, Legion::Runtime *runtime, const std::set<Legion::Processor> &local_procs);

#endif // TACO_MAPPER_H
