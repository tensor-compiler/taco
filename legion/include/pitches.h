#ifndef TACO_PITCHES_H
#define TACO_PITCHES_H

#include "legion.h"

// This is a small helper class that will also work if we have zero-sized arrays
// We also need to have this instead of std::array so that it works on devices.
// This code was taken from https://github.com/nv-legate/legate.numpy/blob/3452c85f93c4a886e9f4bff5f2e87b20f98b30bf/src/point_task.h.
template <int DIM>
class Pitches {
public:
  __CUDA_HD__
  Pitches() {
    for (int i = 0; i < DIM; i++) {
      this->pitches[i] = 0;
    }
  }

  __CUDA_HD__
  inline size_t flatten(const Legion::Rect<DIM + 1>& rect)
  {
    size_t pitch  = 1;
    size_t volume = 1;
    for (int d = DIM; d >= 0; --d) {
      // Quick exit for empty rectangle dimensions
      if (rect.lo[d] > rect.hi[d]) return 0;
      const size_t diff = rect.hi[d] - rect.lo[d] + 1;
      volume *= diff;
      if (d > 0) {
        pitch *= diff;
        pitches[d - 1] = pitch;
      }
    }
    return volume;
  }
  __CUDA_HD__
  inline Legion::Point<DIM + 1> unflatten(size_t index, const Legion::Point<DIM + 1>& lo) const
  {
    Legion::Point<DIM + 1> point = lo;
    for (int d = 0; d < DIM; d++) {
      point[d] += index / pitches[d];
      index = index % pitches[d];
    }
    point[DIM] += index;
    return point;
  }

private:
  size_t pitches[DIM];
};
// Specialization for the zero-sized case
template <>
class Pitches<0> {
public:
  __CUDA_HD__
  inline size_t flatten(const Legion::Rect<1>& rect)
  {
    if (rect.lo[0] > rect.hi[0])
      return 0;
    else
      return (rect.hi[0] - rect.lo[0] + 1);
  }
  __CUDA_HD__
  inline Legion::Point<1> unflatten(size_t index, const Legion::Point<1>& lo) const
  {
    Legion::Point<1> point = lo;
    point[0] += index;
    return point;
  }
};

#endif // TACO_PITCHES_H