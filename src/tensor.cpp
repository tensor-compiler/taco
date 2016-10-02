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

static void packIndices(const vector<size_t>& dims,
                        const vector<vector<int>>& coords,
                        size_t begin, size_t end,
                        const vector<Level>& levels, size_t i,
                        Indices* indices) {
  auto& level = levels[i];
  auto& index = (*indices)[i];
  auto& levelCoords = coords[i];

  switch (level.type) {
    case Level::Dense: {
      // Iterate over each index value
      size_t start = 0;
      for (int j=0; j < (int)dims[i]; ++j) {
        // Scan to find segment range of children
        size_t end = start;
        while (end < levelCoords.size() && levelCoords[end] == j) {
          end++;
        }
        packIndices(dims, coords, start, end, levels, i+1, indices);
        start = end;
      }
      break;
    }
    case Level::Sparse: {
      // Store segment end
      index[0].push_back(end);

      // Iterate over each index value in segment
      index[1].reserve(index[1].size() +
                      countUniqueEntries(levelCoords.begin() + begin,
                                         levelCoords.begin() + end));
      for (size_t j=begin; j < end; ++j) {
        index[1].push_back(levelCoords[j]);
      }
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

        // Add start of first segment
        indices[i][0].push_back(0);
        break;
      }
      case Level::Values: {
        // Do nothing
        break;
      }
    }
  }

  packIndices(dimensions, coords, 0, numCoords, levels, 0, &indices);

  for (size_t i=0; i < indices.size(); ++i) {
    auto& index = indices[i];
    std::cout << "index[" << i << "]" << std::endl;
    for (size_t j=0; j < index.size(); ++j) {
      auto& indexArray = index[j];
      std::cout << "  index[" << i << "][" << j << "] == "
                << "{" << util::join(indexArray) << "}" << std::endl;
    }
  }

  void* vals = malloc(nnz * ctype.bytes());

  return make_shared<PackedTensor>(nnz, vals, indices);
}

}
