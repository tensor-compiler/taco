#include "taco/storage/pack.h"

#include <climits>

#include "taco/format.h"
#include "taco/error.h"
#include "taco/ir/ir.h"
#include "taco/storage/storage.h"
#include "taco/storage/index.h"
#include "taco/storage/array.h"
#include "taco/util/collections.h"

using namespace std;

namespace taco {

#define PACK_NEXT_LEVEL(cend) {                                          \
  if (i + 1 == modeTypes.size()) {                                       \
    if (cbegin < cend) {                                                  \
      memcpy(&values[valuesIndex], &vals[cbegin*dataType.getNumBytes()], dataType.getNumBytes()); \
    }                                                                     \
    else {                                                                 \
      memset(&values[valuesIndex], 0, dataType.getNumBytes());             \
    }                                                                       \
    valuesIndex += dataType.getNumBytes();                                \
  } else {                                                               \
    valuesIndex = packTensor(dimensions, coords, vals, cbegin, (cend), modeTypes, i+1, \
    indices, values, dataType, valuesIndex);                             \
  }                                                                      \
}


/// Count unique entries (assumes the values are sorted)
static TypedIndexVector getUniqueEntries(TypedIndexVector v, 
                                         size_t startIndex, size_t endIndex) {
  TypedIndexVector uniqueEntries(v.getType());
  TypedIndexVal prev;
  TypedIndexVal curr;
  if (endIndex - startIndex > 0){
    prev = v[startIndex];
    uniqueEntries.push_back(prev);
    for (size_t j = startIndex + 1; j < endIndex; j++) {
      curr = v[j];
      taco_iassert(curr >= prev);
      if (curr > prev) {
        prev = curr;
        uniqueEntries.push_back(curr);
      }
    }
  }
  return uniqueEntries;
}

/// Pack tensor coordinates into an index structure and value array.  The
/// indices consist of one index per tensor mode, and each index contains
/// [0,2] index arrays.
static int packTensor(const vector<int>& dimensions,
                      const vector<TypedIndexVector>& coords,
                      char* vals,
                      size_t begin, size_t end,
                      const vector<ModeFormat>& modeTypes, size_t i,
                      std::vector<std::vector<TypedIndexVector>>* indices,
                      char* values, Datatype dataType, int valuesIndex) {
  auto& modeType    = modeTypes[i];
  auto& levelCoords = coords[i];
  auto& index       = (*indices)[i];
  if (modeType.getName() == Dense.getName()) {
    // Iterate over each index value and recursively pack it's segment
    size_t cbegin = begin;
    for (int j=0; j < (int)dimensions[i]; ++j) {
      // Scan to find segment range of children
      size_t cend = cbegin;
      while (cend < end && levelCoords[cend] == j) {
        cend++;
      }
      PACK_NEXT_LEVEL(cend);
      cbegin = cend;
    }
  } else if (modeType.getName() == Sparse.getName()) {
    TypedIndexVector indexValues = getUniqueEntries(levelCoords, begin, end);

    // Store segment end: the size of the stored segment is the number of
    // unique values in the coordinate list
    index[0].push_back((int)index[1].size() + (int)indexValues.size());

    // Store unique index values for this segment
    index[1].push_back_vector(indexValues);

    // Iterate over each index value and recursively pack it's segment
    size_t cbegin = begin;
    for (int j = 0; j < (int) indexValues.size(); j++) {
      // Scan to find segment range of children
      size_t cend = cbegin;
      while (cend < end && levelCoords[cend] == indexValues[j]) {
        cend++;
      }
      PACK_NEXT_LEVEL(cend);
      cbegin = cend;
    }
  } else {
    taco_not_supported_yet;
  }
  return valuesIndex;
}

inline bool sameSize(const std::vector<TypedIndexVector>& coordinates) {
  if (coordinates.size() == 0) return true;
  size_t num = coordinates[0].size();
  for (size_t i = 1; i < coordinates.size(); i++) {
    if (coordinates[i].size() != num) return false;
  }
  return true;
}

/// Pack tensor coordinates into a format. The coordinates must be stored as a
/// structure of arrays, that is one vector per axis coordinate and one vector
/// for the values. The coordinates must be sorted lexicographically.
TensorStorage pack(Datatype                             componentType,
                   const std::vector<int>&              dimensions,
                   const Format&                        format,
                   const std::vector<TypedIndexVector>& coordinates,
                   const void *                         values) {
  taco_iassert(dimensions.size() == (size_t)format.getOrder());
  taco_iassert(coordinates.size() == (size_t)format.getOrder());
  taco_iassert(sameSize(coordinates));
  taco_iassert(dimensions.size() > 0) << "Scalar packing not supported";

  size_t order = dimensions.size();
  size_t numCoordinates = coordinates[0].size();

  TensorStorage storage(componentType, dimensions, format);

  // Create vectors to store pointers to indices/index sizes
  vector<vector<TypedIndexVector>> indices;
  indices.reserve(order);

  long long int maxSize = 1;
  for (size_t i=0; i < order; ++i) {
    ModeFormat modeType = format.getModeFormats()[i];
    if (modeType.getName() == Dense.getName()) {
      indices.push_back({});
      maxSize *= dimensions[i];
    } else if (modeType.getName() == Sparse.getName()) {
      // Sparse indices have two arrays: a segment array and an index array
      indices.push_back({TypedIndexVector(format.getCoordinateTypePos(i)),
                         TypedIndexVector(format.getCoordinateTypeIdx(i))});

      // Add start of first segment
      indices[i][0].push_back(0);

      maxSize = numCoordinates;
    } else {
      taco_not_supported_yet;
    }
  }

  void* vals = malloc(maxSize * componentType.getNumBytes());
  int actual_size = packTensor(dimensions, coordinates, (char *) values, 0,
                               numCoordinates, format.getModeFormats(), 0,
                               &indices, (char *)vals, componentType, 0);
  vals = realloc(vals, actual_size);

  // Create a tensor index
  vector<ModeIndex> modeIndices;
  for (size_t i = 0; i < order; i++) {
    ModeFormat modeType = format.getModeFormats()[i];
    if (modeType.getName() == Dense.getName()) {
      Array size = makeArray({dimensions[i]});
      modeIndices.push_back(ModeIndex({size}));
    } else if (modeType.getName() == Sparse.getName()) {
      Array pos = makeArray(format.getCoordinateTypePos(i),indices[i][0].size());
      memcpy(pos.getData(), indices[i][0].data(),
             indices[i][0].size()*format.getCoordinateTypePos(i).getNumBytes());

      Array idx = makeArray(format.getCoordinateTypeIdx(i), indices[i][1].size());
      memcpy(idx.getData(), indices[i][1].data(), indices[i][1].size() * format.getCoordinateTypeIdx(i).getNumBytes());
      modeIndices.push_back(ModeIndex({pos, idx}));
    } else {
      taco_not_supported_yet;
    }
  }
  storage.setIndex(Index(format, modeIndices));
  Array array = makeArray(componentType,
                          actual_size/componentType.getNumBytes());
  memcpy(array.getData(), vals, actual_size);
  storage.setValues(array);
  free(vals);
  return storage;
}

}
