#include "tensor.h"

#include "packed_tensor.h"
#include "format.h"
#include "tree.h"

using namespace std;

namespace taco {

typedef PackedTensor::IndexType  IndexType;
typedef PackedTensor::IndexArray IndexArray;
typedef PackedTensor::Index      Index;
typedef PackedTensor::Indices    Indices;


/// Count unique entries between iterators
static size_t countUniqueEntries(const vector<int>::const_iterator& begin,
                                 const vector<int>::const_iterator& end) {
  size_t uniqueEntries = 0;

  // Assumes sorting
  if (begin != end) {
    uniqueEntries = 1;
    size_t curr = *begin;
    for (auto it = begin+1; it != end; ++it) {
      size_t next = *it;
      iassert(next >= curr);
      if (next > curr) {
        uniqueEntries++;
        curr = next;
      }
    }
  }
  return uniqueEntries;
}

static void computeIndexSizes(const vector<vector<int>>& coords,
                              size_t first, size_t last,
                              const vector<Level>& levels, size_t i,
                              Indices* indices) {
  auto& level = levels[i];
  auto& index = (*indices)[i];
  auto& levelCoords = coords[i];

  switch (level.type) {
    case Level::Dense: {
      // Do Nothing
      break;
    }
    case Level::Sparse: {
      index[0].first  += 1;
      index[1].first += countUniqueEntries(levelCoords.begin(),
                                           levelCoords.end());
      break;
    }
    case Level::Values: {
      // Do nothing
      break;
    }
  }
}

shared_ptr<PackedTensor>
pack(const vector<size_t>& dimensions, internal::ComponentType ctype,
     const Format& format, const vector<vector<int>>& coords,
     const void* values) {
  iassert(coords.size() > 0);
  size_t numCoords = coords[0].size();
  std::cout << "numCoords: " << numCoords << std::endl;

  const vector<Level>& levels = format.getLevels();
  Indices indices(levels.size());

  // Create the vectors to store pointers to indices/index sizes
  size_t nnz = 1;
  for (size_t i=0; i < levels.size(); ++i) {
    auto& level = levels[i];
    switch (level.type) {
      case Level::Dense: {
        nnz *= dimensions[i];
        break;
      }
      case Level::Sparse: {
        // A sparse level packs nnz down to #coords
        nnz = numCoords;

        // Sparse indices have two arrays: a segment array and an index array
        indices[i].resize(2);

        // Add space for sentinel
        indices[i][0].first = 1;
        break;
      }
      case Level::Values: {
        break;  // Do nothing
      }
    }
  }

  computeIndexSizes(coords, 0, numCoords, levels, 0, &indices);

  // Allocate index memory
  for (auto& index : indices) {
    for (auto& indexArray : index) {
      size_t size = indexArray.first;
      indexArray.second = (IndexType*)malloc(size * sizeof(IndexType));
    }
  }

  for (size_t i=0; i < indices.size(); ++i) {
    auto& index = indices[i];
    std::cout << "index[" << i << "].size() == " << index.size() << std::endl;
    for (size_t j=0; j < index.size(); ++j) {
      auto& indexArray = index[j];
      std::cout << "  index[" << i << "][" << j << "] == " << indexArray.first
                << std::endl;
    }
  }

  void* vals = malloc(nnz * ctype.bytes());

  return make_shared<PackedTensor>(nnz, vals, indices);
}

}
