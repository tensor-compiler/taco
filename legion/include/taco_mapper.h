#ifndef TACO_MAPPER_H
#define TACO_MAPPER_H

#include "legion.h"
#include "mappers/default_mapper.h"

// Register the TACO mapper.
void register_taco_mapper(Legion::Machine machine, Legion::Runtime *runtime, const std::set<Legion::Processor> &local_procs);

class TACOMapper : public Legion::Mapping::DefaultMapper {
public:
  TACOMapper(Legion::Mapping::MapperRuntime *rt, Legion::Machine& machine, const Legion::Processor& local)
      : DefaultMapper(rt, machine, local) {}

  void default_policy_select_constraints(Legion::Mapping::MapperContext ctx,
                                         Legion::LayoutConstraintSet &constraints, Legion::Memory target_memory,
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

  void default_policy_select_target_processors(
      Legion::Mapping::MapperContext ctx,
      const Legion::Task &task,
      std::vector<Legion::Processor> &target_procs) {
    // TODO (rohany): Add a TACO tag to the tasks.
    if (task.is_index_space) {
      // Index launches should be placed directly on the processor
      // they were sliced to.
      target_procs.push_back(task.target_proc);
    } else if (std::string(task.get_task_name()).find("task_") != std::string::npos) {
      // Other point tasks should stay on the originating processor.
      target_procs.push_back(task.orig_proc);
    } else {
      DefaultMapper::default_policy_select_target_processors(ctx, task, target_procs);
    }
  }

  // TODO (rohany): A strategy for slicing so that we can place data on faces:
  //  * Make a PointInRectIterator which iterates over the full processor grid
  //    desired by the data placement.
  //  * Don't assign processors with point values != to the face value.
  //  * The last problem here is getting the face information over to the mapper.
};

#endif // TACO_MAPPER_H
