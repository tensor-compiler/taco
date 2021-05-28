#ifndef TACO_MAPPER_H
#define TACO_MAPPER_H

#include "legion.h"
#include "mappers/default_mapper.h"

const Legion::ShardingID TACOShardingFunctorID = 15210;

// Register the TACO mapper.
void register_taco_mapper(Legion::Machine machine, Legion::Runtime *runtime, const std::set<Legion::Processor> &local_procs);

class TACOMapper : public Legion::Mapping::DefaultMapper {
public:
  TACOMapper(Legion::Mapping::MapperRuntime *rt, Legion::Machine& machine, const Legion::Processor& local)
      : DefaultMapper(rt, machine, local) {}

  // Mapping tags handled specific for the TACO mapper.
  enum MappingTags {
    // Indicates that this task launch is used for data placement.
    PLACEMENT = (1 << 5),
  };

  void select_sharding_functor(const Legion::Mapping::MapperContext ctx,
                               const Legion::Task& task,
                               const SelectShardingFunctorInput& input,
                               SelectShardingFunctorOutput& output) override {
    // Use TACO's sharding functor instead of the default sharding functor.
    output.chosen_functor = TACOShardingFunctorID;
  }

  void default_policy_select_constraints(Legion::Mapping::MapperContext ctx,
                                         Legion::LayoutConstraintSet &constraints, Legion::Memory target_memory,
                                         const Legion::RegionRequirement &req) override {
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

  void default_policy_select_target_processors(
      Legion::Mapping::MapperContext ctx,
      const Legion::Task &task,
      std::vector<Legion::Processor> &target_procs) override {
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

  std::vector<Legion::Processor> select_targets_for_task(const Legion::Mapping::MapperContext ctx,
                                                         const Legion::Task& task) {
    auto kind = this->default_find_preferred_variant(task, ctx, false /* needs tight bounds */).proc_kind;
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
  }

  void slice_task(const Legion::Mapping::MapperContext    ctx,
                  const Legion::Task&                     task,
                  const SliceTaskInput&                   input,
                  SliceTaskOutput&                        output) override {
    if (task.tag & PLACEMENT) {
      // Placement tasks will put the dimensions of the placement grid at the beginning
      // of the task arguments. Here, we extract the packed placement grid dimensions.
      int dim = input.domain.get_dim();
      int* args = (int*)(task.args);
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

public:
  // select_num_blocks is a simple wrapper that allows external access to the
  // DefaultMapper's protected default_select_num_blocks method.
  template<int DIM>
  static Legion::Point<DIM,Legion::coord_t> select_num_blocks(
      long long int factor,
      const Legion::Rect<DIM,Legion::coord_t> &rect_to_factor) {
    return DefaultMapper::default_select_num_blocks(factor, rect_to_factor);
  }
  // TODO (rohany): It may end up being necessary that we need to explicitly map
  //  regions for placement tasks. If so, Manolis says the following approach
  //  is the right thing:
  //  * Slice tasks sends tasks where they are supposed to go.
  //  * Map task needs to create a new instance of the region visible to the
  //    target processor.
};

class TACOShardingFunctor : public Legion::ShardingFunctor {
public:
  virtual Legion::ShardID shard(const Legion::DomainPoint& point,
                                const Legion::Domain& launch_space,
                                const size_t total_shards) {
    // This sharding functor attempts to perform a similar block-wise decomposition
    // that the default mapper performs when slicing a task. It is equivalent to
    // the default sharding functor when the number of shards is equal to the
    // number of points in the launch space.
    switch (launch_space.dim) {
#define BLOCK(DIM) \
        case DIM:  \
          {        \
            auto launchRect = Legion::Rect<DIM>(launch_space); \
            /* Require that all index spaces start at 0. */    \
            for (int i = 0; i < (DIM); i++) {                    \
              assert(launchRect.lo[i] == 0);                   \
            }      \
            auto blocks = TACOMapper::select_num_blocks<DIM>(total_shards, launchRect); \
            Legion::Point<DIM> zeroes, ones;                   \
            for (int i = 0; i < (DIM); i++) {                    \
                zeroes[i] = 0;                                 \
                ones[i] = 1;                                   \
            }      \
            Legion::Rect<DIM> blockSpace(zeroes, blocks - ones);       \
            auto numPoints = launchRect.hi - launchRect.lo + ones;                      \
            /* Invert the block -> point computation in default_decompose_points. */    \
            Legion::Point<DIM> projected;                      \
            for (int i = 0; i < (DIM); i++) {                    \
              projected[i] = point[i] * blocks[i] / numPoints[i];                       \
            }      \
            Realm::AffineLinearizedIndexSpace<DIM, Legion::coord_t> linearizer(blockSpace); \
            return linearizer.linearize(projected);            \
          }
      LEGION_FOREACH_N(BLOCK)
#undef BLOCK
      default:
        assert(false);
    }
  }
};

#endif // TACO_MAPPER_H
