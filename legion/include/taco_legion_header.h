#ifndef TACO_LEGION_INCLUDES_H
#define TACO_LEGION_INCLUDES_H

#include "legion.h"
#include "mappers/default_mapper.h"

// Field used by the generated TACO code.
enum TensorFields {
  FID_VAL
};
const int TACO_TASK_BASE_ID = 1000;
const int TACO_SHARD_BASE_ID = 1000;

Legion::IndexSpace get_index_space(Legion::PhysicalRegion r);
Legion::IndexSpace get_index_space(Legion::LogicalRegion r);
Legion::LogicalRegion get_logical_region(Legion::PhysicalRegion r);
Legion::LogicalRegion get_logical_region(Legion::LogicalRegion r);
Legion::IndexPartition get_index_partition(Legion::IndexPartition i);
Legion::IndexPartition get_index_partition(Legion::LogicalPartition l);
int getIndexPoint(const Legion::Task* task, int index);
Legion::TaskID taskID(int offset);
Legion::ShardingID shardingID(int offset);

void registerPlacementShardingFunctor(Legion::Context ctx, Legion::Runtime* runtime, Legion::ShardingID funcID, std::vector<int>& dims);

#endif // TACO_LEGION_INCLUDES_H
