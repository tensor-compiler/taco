#include "taco/storage/pack.h"

#include <climits>

#include "taco/format.h"
#include "taco/error.h"
#include "taco/ir/ir.h"
#include "taco/storage/storage.h"
#include "taco/storage/index.h"
#include "taco/storage/array.h"
#include "taco/storage/array_util.h"
#include "taco/util/collections.h"

using namespace std;

namespace taco {
namespace storage {

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

void pushToCharVector(vector<char> *v, void *value, size_t size = sizeof(int)) {
  (*v).resize((*v).size() + size);
  memcpy(&(*v)[(*v).size() - size], value, size);
}


/// Count unique entries (assumes the values are sorted)
vector<char> getUniqueEntries(vector<char> v, int startIndex, int endIndex) {
  int uniqueCount = 0;
  vector<char> uniqueEntries;
  if (endIndex - startIndex > 0){
    int prev = *(((int *) v.data()) + startIndex);
    pushToCharVector(&uniqueEntries, &prev);
    uniqueCount++;
    for (int *j = ((int *) v.data()) + 1 + startIndex; j < ((int *) v.data()) + endIndex; j++) {
      int curr = *j;
      taco_iassert(curr >= prev);
      if (curr > prev) {
        prev = curr;
        pushToCharVector(&uniqueEntries, &curr);
        uniqueCount++;
      }
    }
  }
  return uniqueEntries;
}





size_t findMaxFixedValue(const vector<int>& dimensions,
                              const vector<vector<char>>& coords,
                              size_t order,
                              const size_t fixedLevel,
                              const size_t i, const size_t numCoords) {
  if (i == order) {
    return numCoords;
  }
  if (i == fixedLevel) {
    auto indexValues = getUniqueEntries(coords[i], 0, coords[i].size()/sizeof(int));
    return indexValues.size() / sizeof(int);
  }
  else {
    // Find max occurrences for level i
    size_t maxSize=0;
    vector<char> maxCoords;
    int coordCur= *((int *) &(coords[i][0]));
    size_t sizeCur=1;
    for (size_t j=1; j<numCoords; j++) {
      if (memcmp(&coords[i][j*sizeof(int)], &coordCur, sizeof(int)) == 0) {
        sizeCur++;
      }
      else {
        if (sizeCur > maxSize) {
          maxSize = sizeCur;
          maxCoords.clear();
          pushToCharVector(&maxCoords, &coordCur);
        }
        else if (sizeCur == maxSize) {
          pushToCharVector(&maxCoords, &coordCur);
        }
        sizeCur=1;
        coordCur=*((int *) &(coords[i][j*sizeof(int)]));
      }
    }
    if (sizeCur > maxSize) {
      maxSize = sizeCur;
      maxCoords.clear();
      pushToCharVector(&maxCoords, &coordCur);
    }
    else if (sizeCur == maxSize) {
      pushToCharVector(&maxCoords, &coordCur);
    }

    size_t maxFixedValue=0;
    size_t maxSegment;
    vector<vector<char>> newCoords(order);
    for (size_t l=0; l<maxCoords.size()/sizeof(int); l++) {
      // clean coords for next level
      for (size_t k=0; k<order;k++) {
        newCoords[k].clear();
      }
      for (size_t j=0; j<numCoords; j++) {
        if (memcmp(&coords[i][j*sizeof(int)], &maxCoords[l*sizeof(int)], sizeof(int)) == 0) {
          for (size_t k=0; k<order;k++) {
            pushToCharVector(&newCoords[k], (int *) &coords[k][j*sizeof(int)]);
          }
        }
      }
      maxSegment = findMaxFixedValue(dimensions, newCoords, order, fixedLevel,
                                     i+1, maxSize);
      maxFixedValue = std::max(maxFixedValue,maxSegment);
    }
    return maxFixedValue;
  }
}

/// Pack tensor coordinates into an index structure and value array.  The
/// indices consist of one index per tensor mode, and each index contains
/// [0,2] index arrays.
int packTensor(const vector<int>& dimensions,
                const vector<vector<char>>& coords,
                char* vals,
                size_t begin, size_t end,
                const vector<ModeType>& modeTypes, size_t i,
                std::vector<std::vector<std::vector<char>>>* indices,
                char* values, DataType dataType, int valuesIndex) {
  auto& modeType    = modeTypes[i];
  auto& levelCoords = coords[i];
  auto& index       = (*indices)[i];

  switch (modeType) {
    case Dense: {
      // Iterate over each index value and recursively pack it's segment
      size_t cbegin = begin;
      for (int j=0; j < (int)dimensions[i]; ++j) {
        // Scan to find segment range of children
        size_t cend = cbegin;
        while (cend < end && memcmp(&levelCoords[cend*sizeof(int)], &j, sizeof(int)) == 0) {
          cend++;
        }
        PACK_NEXT_LEVEL(cend);
        cbegin = cend;
      }
      break;
    }
    case Sparse: {
      auto indexValues = getUniqueEntries(levelCoords, begin, end);

      // Store segment end: the size of the stored segment is the number of
      // unique values in the coordinate list
      int value = (index[1].size() + indexValues.size()) / sizeof(int);
      pushToCharVector(&index[0], &value);

      // Store unique index values for this segment
      /*index[1].resize(index[1].size() + indexValues.size());
      memcpy(&index[1][index[1].size() - indexValues.size()], indexValues.data(), indexValues.size());*/
      index[1].insert(index[1].end(), indexValues.begin(), indexValues.end());

      // Iterate over each index value and recursively pack it's segment
      size_t cbegin = begin;
      for (int *j = (int *) indexValues.data(); j < (int *) (indexValues.data() + indexValues.size()); j++) {
        // Scan to find segment range of children
        size_t cend = cbegin;
        while (cend < end && memcmp(&levelCoords[cend*sizeof(int)], j, sizeof(int)) == 0) {
          cend++;
        }
        PACK_NEXT_LEVEL(cend);
        cbegin = cend;
      }
      break;
    }
    case Fixed: {
      size_t fixedValue = index[0][0];
      auto indexValues = getUniqueEntries(levelCoords, begin, end);

      // Store segment end: the size of the stored segment is the number of
      // unique values in the coordinate list
      size_t segmentSize = indexValues.size() / sizeof(int);
      // Store unique index values for this segment
      size_t cbegin = begin;
      if (segmentSize > 0) {
        /*index[1].resize(index[1].size() + indexValues.size());
        memcpy(&index[1][index[1].size() - indexValues.size()], indexValues.data(), indexValues.size());*/
        index[1].insert(index[1].end(), indexValues.begin(), indexValues.end());
        for (int *j = (int *) indexValues.data(); j < (int *) (indexValues.data() + indexValues.size()); j++) {
          // Scan to find segment range of children
          size_t cend = cbegin;
          while (cend < end && memcmp(&levelCoords[cend*sizeof(int)], j, sizeof(int)) == 0) {
            cend++;
          }
          PACK_NEXT_LEVEL(cend);
          cbegin = cend;
        }
      }
      // Complete index if necessary with the last index value
      auto curSize=segmentSize;
      while (curSize < fixedValue) {
        int value = (segmentSize > 0) ? indexValues[(segmentSize-1)*sizeof(int)] : 0;
        pushToCharVector(&index[1], &value);
        PACK_NEXT_LEVEL(cbegin);
        curSize++;
      }
      break;
    }
  }
  return valuesIndex;
}
  
/// Pack tensor coordinates into a format. The coordinates must be stored as a
/// structure of arrays, that is one vector per axis coordinate and one vector
/// for the values. The coordinates must be sorted lexicographically.
Storage pack(const std::vector<int>&              dimensions,
             const Format&                        format,
             const std::vector<std::vector<char>>& coordinates,
             const void *            values,
             const size_t numCoordinates,
             DataType datatype, DataType coordType) {
  taco_iassert(dimensions.size() == format.getOrder());

  Storage storage(format);

  size_t order = dimensions.size();

  // Create vectors to store pointers to indices/index sizes
  vector<vector<vector<char>>> indices;
  indices.reserve(order);

  for (size_t i=0; i < order; ++i) {
    switch (format.getModeTypes()[i]) {
      case Dense: {
        indices.push_back({});
        break;
      }
      case Sparse: {
        // Sparse indices have two arrays: a segment array and an index array
        indices.push_back({{}, {}});

        // Add start of first segment
        long long val = 0;
        pushToCharVector(&indices[i][0], &val, coordType.getNumBytes());
        break;
      }
      case Fixed: {
        // Fixed indices have two arrays: a segment array and an index array
        indices.push_back({{}, {}});

        // Add maximum size to segment array
        long long maxSize = findMaxFixedValue(dimensions, coordinates,
                                           format.getOrder(), i, 0,
                                           numCoordinates);
        taco_iassert(maxSize <= INT_MAX);
        pushToCharVector(&indices[i][0], &maxSize, coordType.getNumBytes()); //TODO is this right?

        break;
      }
    }
  }

  int max_size = 1;
  for (int i : dimensions)
    max_size *= i;

  void* vals = malloc(max_size * datatype.getNumBytes()); //has zeroes where dense
  int actual_size = packTensor(dimensions, coordinates, (char *) values, 0,
             numCoordinates, format.getModeTypes(), 0, &indices, (char *) vals, datatype, 0);

  vals = realloc(vals, actual_size);

  // Create a tensor index
  vector<ModeIndex> modeIndices;
  for (size_t i = 0; i < order; i++) {
    ModeType modeType = format.getModeTypes()[i];
    switch (modeType) {
      case ModeType::Dense: {
        Array size = makeArray({dimensions[i]});
        modeIndices.push_back(ModeIndex({size}));
        break;
      }
      case ModeType::Sparse:
      case ModeType::Fixed: {
        Array pos = makeArray(Int(), indices[i][0].size() / coordType.getNumBytes());
        memcpy(pos.getData(), indices[i][0].data(), indices[i][0].size());

        Array idx = makeArray(Int(), indices[i][1].size() / coordType.getNumBytes());
        memcpy(idx.getData(), indices[i][1].data(), indices[i][1].size());
        modeIndices.push_back(ModeIndex({pos, idx}));
        break;
      }
    }
  }
  storage.setIndex(Index(format, modeIndices));
  Array array = makeArray(datatype, actual_size/datatype.getNumBytes());
  memcpy(array.getData(), vals, actual_size);
  storage.setValues(array);
  free(vals);
  return storage;
}


ir::Stmt packCode(const Format& format) {
  using namespace taco::ir;

  vector<Stmt> packStmts;

  // Generate loops to count the size of each level.
//  ir::Stmt countLoops;
//  for (auto& level : util::reverse(format.getLevels())) {
//    if (countLoops.defined()) {
//
//    }
//  }

  // Loops to insert index values
  Stmt insertLoop;
  for (ModeType modeType : util::reverse(format.getModeTypes())) {
    Stmt body = insertLoop.defined()
        ? insertLoop
        : VarAssign::make(Var::make("test", Int()), 1.0, true);

    switch (modeType) {
      case Dense: {
        Expr dimension = (long long) 10;
        Expr loopVar = Var::make("i", Int());
        insertLoop = ir::For::make(loopVar, (long long) 0, dimension, (long long) 1, body);
        break;
      }
      case Sparse: {
        break;
      }
      case Fixed: {
        taco_not_supported_yet;
        break;
      }
    }
  }
  packStmts.push_back(insertLoop);

  return ir::Block::make(packStmts);
}

}}
