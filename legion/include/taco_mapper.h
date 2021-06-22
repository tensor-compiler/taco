#ifndef TACO_MAPPER_H
#define TACO_MAPPER_H

#include "legion.h"
#include "mappers/default_mapper.h"

#include "task_ids.h"
#include "shard.h"

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
