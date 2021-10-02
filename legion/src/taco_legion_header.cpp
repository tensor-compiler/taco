#include "taco_legion_header.h"
#include "shard.h"
#include <mutex>

using namespace Legion;

IndexSpace get_index_space(PhysicalRegion r) { return r.get_logical_region().get_index_space(); }
IndexSpace get_index_space(LogicalRegion r) { return r.get_index_space(); }

LogicalRegion get_logical_region(PhysicalRegion r) { return r.get_logical_region(); }
LogicalRegion get_logical_region(LogicalRegion r) { return r; }

IndexPartition get_index_partition(IndexPartition i) { return i; }
IndexPartition get_index_partition(LogicalPartition l) { return l.get_index_partition(); }

int getIndexPoint(const Legion::Task* task, int index) {
  return task->index_point[index];
}

TaskID taskID(int offset) {
  return TACO_TASK_BASE_ID + offset;
}

ShardingID shardingID(int offset) {
  return TACO_SHARD_BASE_ID + offset;
}

void registerPlacementShardingFunctor(Context ctx, Runtime* runtime, ShardingID funcID, std::vector<int>& dims) {
  // If we have multiple shards on the same node, they might all try to register sharding
  // functors at the same time. Put a lock here to make sure that only one actually does it.
  static std::mutex mu;
  std::lock_guard<std::mutex> lock(mu);

  auto func = Legion::Runtime::get_sharding_functor(funcID);
  // If the sharding functor already exists, return.
  if (func) { return; }
  // Otherwise, register the functor.
  auto functor = new TACOPlacementShardingFunctor(dims);
  runtime->register_sharding_functor(funcID, functor, true /* silence_warnings */);
}

//Color partitionColor(Legion::DomainPoint& dims, Legion::DomainPoint& point) {
//  assert(dims.dim == point.dim);
//  switch (dims.dim) {
//#define BLOCK(DIM) \
//      case DIM: {  \
//        Point<DIM> strides; \
//        for (int i = 0; i < (DIM); i++) { \
//          strides[i] = 0;           \
//        }          \
//        strides[(DIM) - 1] = 1;           \
//        for (int i = (DIM) - 2; i >= 0; i--) { \
//          strides[i] = strides[i + 1] * dims[i]; \
//        }          \
//        Color result = 0;   \
//        for (int i = 0; i < (DIM); i++) { \
//          result += strides[i] + point[i];           \
//        }          \
//        return result;      \
//      }
//#undef BLOCK
//    default:
//      assert(false);
//  };
//}

// TODO (rohany): Test this.
// TODO (rohany): This doesnt work because each task's launch domain
//  is of a different size. So I can't just linearize the points in
//  each launch domain because they all will collide?
// TODO (rohany): Maybe this actually works though? I don't need each
//  depth slice to be "equally sized" because for a given depth the
//  color ID's are all in a different namespace!
//Color constructPartitionColor(TACOPartitionColors kind, int32_t taskDepth, Legion::DomainPoint dims, Legion::DomainPoint point) {
//  // We'll make this 5 for now. We'll assert no one is doing more
//  // than 3-D index launches.
//  size_t strides[5];
//  assert(point.dim <= 3);
//  // The total number of entries we'll use in strides is 2 (kind, taskDepth)
//  // plus the number of points in the launch domain.
//  int totDim = 2 + point.dim;
//
//  for (int i = 0; i < totDim; i++) {
//    strides[i] = 0;
//  }
//  strides[totDim - 1] = 1;
//  for (int i = totDim - 2; i >= 0; i--) {
//    if (i >= 2) {
//      strides[i] = strides[i + 1] * dims[i - 2];
//    } else if (i == 1) {
//      strides[i] = strides[i + 1] * TACO_MAX_TASK_DEPTH;
//    } else if (i == 0) {
//      strides[i] = strides[i + 1] * PID_NUM_PIDS;
//    }
//  }
//  Color flattened = 0;
//  flattened += kind * strides[0];
//  flattened += taskDepth * strides[1];
//  for (int i = 2; i < totDim; i++) {
//    flattened += point[i - 2] * strides[i];
//  }
//  return flattened;
//
////  Color result;
////  result[0] = kind;
////  result[1] = taskDepth;
////
////  // Construct a strides domain point.
////  DomainPoint strides = dims;
////  for (int i = 0; i < dims.dim; i++) {
////    strides[i] = 0;
////  }
////  strides[dims.dim - 1] = 1;
////  for (int i = dims.dim - 2; i >= 0; i--) {
////    strides[i] = strides[i + 1] * dims[i];
////  }
////
////  // Flatten the point.
////  size_t flattened = 0;
////  assert(dims.dim == point.dim);
////  for (int i = 0; i < dims.dim; i++) {
////    flattened += point[i] * strides[i];
////  }
////  result[2] = flattened;
////  return result;
//}
