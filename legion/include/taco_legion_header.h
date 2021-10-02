#ifndef TACO_LEGION_INCLUDES_H
#define TACO_LEGION_INCLUDES_H

#include "legion.h"
#include "mappers/default_mapper.h"

// Field used by the generated TACO code.
enum TensorFields {
  FID_VAL
};
const int TACO_TASK_BASE_ID = 10000;
const int TACO_SHARD_BASE_ID = 1000;
const int TACO_MAX_TASK_DEPTH = 20;

// Identifier for partitions created by TACO for placement and computation.
// TODO (rohany): See if we need this declaration to be visible elsewhere (or I guess
//  the main file can just import this?).
enum TACOPartitionColors {
  PID_PLACEMENT = 1000,
  PID_COMPUTE,
};

// TODO (rohany): Let's try something simpler:
//  Once the top level partition is created, it doesn't need to be identified
//  anymore with the different partition IDs, because all subpartitions of each
//  are in separate namespaces. Additionally, each task depth / partition level
//  is also in a separate namespace. Therefore, at each namespace, we only care
//  about the index domain that we're creating the partition within, so we can
//  just linearize within that.
// TODO (rohany): Maybe this doesn't work? The case I'm worried about is when we
//  do an index launch over a region (no part), and then do another index launch over
//  the region again within each leaf task. Then, each leaf task creates a partition.
//  In this case, we would create a bunch of partitions of the region all with the same
//  domain, when instead each of the child partitions should have a unique ID. One
//  thing to do is to directly make the color of the partition dependent on the value
//  of each index variable on the path of the task up to the root.
//Legion::Color partitionColor(Legion::DomainPoint& dims, Legion::DomainPoint& point);


// TODO (rohany): This doesn't work / is too complicated.
// constructPartitionColor constructs a 3-D domain point that identifies a partition. It is
// a tuple of the partition kind (placement or compute), the depth in the task tree that the
// partition was created, and a linearized version of loop iterator at the loop level the
// partition is being created at.
// TODO (rohany): Update comment. Basically, we'll have to also create a space
//  with the kind and the the task depth linearized.
//Legion::Color constructPartitionColor(TACOPartitionColors kind, int32_t taskDepth, Legion::DomainPoint dims, Legion::DomainPoint point);

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
