#ifndef TACO_SHARD_H
#define TACO_SHARD_H

#include <mutex>
#include "legion.h"
#include "mappers/default_mapper.h"

// ID for the default taco sharding functor.
const Legion::ShardingID TACOShardingFunctorID = 15210;

// ShimMapper exists just to allow public access to the default_select_blocks
// static method in the DefaultMapper.
class ShimMapper : public Legion::Mapping::DefaultMapper {
public:
  // select_num_blocks is a simple wrapper that allows external access to the
  // DefaultMapper's protected default_select_num_blocks method.
  template<int DIM>
  static Legion::Point<DIM,Legion::coord_t> select_num_blocks(
      long long int factor,
      const Legion::Rect<DIM,Legion::coord_t> &rect_to_factor) {
    return DefaultMapper::default_select_num_blocks(factor, rect_to_factor);
  }
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
            auto blocks = ShimMapper::select_num_blocks<DIM>(total_shards, launchRect); \
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

// TACOPlacementShardingFunctor is a sharding functor that is intended to
// be used by Placement operations. The fact that the placement grid dimensions
// may not be known until run time complicates this a good amount. In particular,
// the placement code must instantiate a sharding functor specific to the placement
// code before performing the index launch that does the placement operation.
// TODO (rohany): A strategy for this is as follows:
//  * Code generation needs to assign a static ID to a sharding functor for
//    every placement operation in the program.
//  * [DONE] When a tag is applied to an index launch, the mapper (in select_sharding_functor)
//    will read this tag and dispatch to special logic. Otherwise, it will return
//    the normal sharding functor.
//  * [DONE] The special logic uses Legion::Runtime::get_sharding_functor() to get the sharding
//    functor with the functor ID in the payload. If the sharding functor requested exists
//    already, then the ID passed through is used. Otherwise, the mapper must create the
//    desired sharding functor and register it. It will extract out the grid dimensions
//    payload, and create the sharding functor with the desired grid dimensions.
class TACOPlacementShardingFunctor : public Legion::ShardingFunctor {
public:
  TACOPlacementShardingFunctor(std::vector<int> gridDims) : gridDims(gridDims) {}
  virtual Legion::ShardID shard(const Legion::DomainPoint& point,
                                const Legion::Domain& launch_space,
                                const size_t total_shards) {
    // Take a lock on the mutex for this functor.
    std::lock_guard<std::mutex> guard(this->mu);
    // If the cache is empty, fill the cache with all point entries.
    // This ensures that we iterate over the launch space once, and
    // use cached results for all other points in the space.
    if (this->cache.size() == 0) {
      switch (launch_space.dim) {
#define BLOCK(DIM) \
        case DIM: {  \
          fillCache<DIM>(launch_space, total_shards); \
          break;   \
        }
        LEGION_FOREACH_N(BLOCK)
#undef BLOCK
        default:
          assert(false);
      }
    }
    assert(this->cachedSpace == launch_space);
    return this->cache.at(point);
  }
private:
  // Fill the precomputed cache of all of the shard positions. This
  // requires that mu is locked.
  template<int DIM>
  void fillCache(const Legion::Domain& launch_space, int total_shards) {
    assert(DIM == this->gridDims.size());
    this->cachedSpace = launch_space;
    // We'll iterate over all of the shards in the placement rect.
    Legion::Rect<DIM> procRect;
    for (int i = 0; i < DIM; i++) {
      procRect.lo[i] = 0;
      procRect.hi[i] = gridDims[i] - 1;
    }
    auto shard = 0;
    for (Legion::PointInRectIterator<DIM> itr(procRect); itr(); itr++) {
      // Always increment the shard counter, so that we skip nodes when we
      // have a Face() placement restriction.
      auto curShard = shard++;
      if (!launch_space.contains(*itr)) {
        continue;
      }
      this->cache[*itr] = curShard % total_shards;
    }
  }
  std::vector<int> gridDims;

  // A mutex protected cache of point -> shard allocations.
  std::mutex mu;
  std::map<Legion::DomainPoint, Legion::ShardID> cache;
  Legion::Domain cachedSpace;
};
#endif // TACO_SHARD_H
