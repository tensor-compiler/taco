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

/// Count unique entries (assumes the values are sorted)
static vector<int> getUniqueEntries(const vector<int>::const_iterator& begin,
                                    const vector<int>::const_iterator& end) {
  vector<int> uniqueEntries;
  if (begin != end) {
    int curr = *begin;
    uniqueEntries.push_back(curr);
    for (auto it = begin+1; it != end; ++it) {
      int next = *it;
      taco_iassert(next >= curr);
      if (curr < next) {
        curr = next;
        uniqueEntries.push_back(curr);
      }
    }
  }
  return uniqueEntries;
}

#define PACK_NEXT_LEVEL(cend) {                                            \
    if (i + 1 == modeTypes.size()) {                                       \
      values->push_back((cbegin < cend) ? vals[cbegin] : 0.0);             \
    } else {                                                               \
      packTensor(dimensions, coords, vals, cbegin, (cend), modeTypes, i+1, \
                 indices, values);                                         \
    }                                                                      \
}

/// Pack tensor coordinates into an index structure and value array.  The
/// indices consist of one index per tensor mode, and each index contains
/// [0,2] index arrays.
static void packTensor(const vector<int>& dimensions,
                       const vector<vector<int>>& coords,
                       const double* vals,
                       size_t begin, size_t end,
                       const vector<ModeType>& modeTypes, size_t i,
                       std::vector<std::vector<std::vector<int>>>* indices,
                       vector<double>* values) {
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
        while (cend < end && levelCoords[cend] == j) {
          cend++;
        }
        PACK_NEXT_LEVEL(cend);
        cbegin = cend;
      }
      break;
    }
    case Sparse: {
      auto indexValues = getUniqueEntries(levelCoords.begin()+begin,
                                          levelCoords.begin()+end);

      // Store segment end: the size of the stored segment is the number of
      // unique values in the coordinate list
      index[0].push_back((int)(index[1].size() + indexValues.size()));

      // Store unique index values for this segment
      index[1].insert(index[1].end(), indexValues.begin(), indexValues.end());

      // Iterate over each index value and recursively pack it's segment
      size_t cbegin = begin;
      for (size_t j : indexValues) {
        // Scan to find segment range of children
        size_t cend = cbegin;
        while (cend < end && levelCoords[cend] == (int)j) {
          cend++;
        }
        PACK_NEXT_LEVEL(cend);
        cbegin = cend;
      }
      break;
    }
    case Fixed: {
      size_t fixedValue = index[0][0];
      auto indexValues = getUniqueEntries(levelCoords.begin()+begin,
                                          levelCoords.begin()+end);

      // Store segment end: the size of the stored segment is the number of
      // unique values in the coordinate list
      size_t segmentSize = indexValues.size() ;
      // Store unique index values for this segment
      size_t cbegin = begin;
      if (segmentSize > 0) {
        index[1].insert(index[1].end(), indexValues.begin(), indexValues.end());
        for (size_t j : indexValues) {
          // Scan to find segment range of children
          size_t cend = cbegin;
          while (cend < end && levelCoords[cend] == (int)j) {
            cend++;
          }
          PACK_NEXT_LEVEL(cend);
          cbegin = cend;
        }
      }
      // Complete index if necessary with the last index value
      auto curSize=segmentSize;
      while (curSize < fixedValue) {
        index[1].insert(index[1].end(), 
                        (segmentSize > 0) ? indexValues[segmentSize-1] : 0);
        PACK_NEXT_LEVEL(cbegin);
        curSize++;
      }
      break;
    }
  }
}

static size_t findMaxFixedValue(const vector<int>& dimensions,
                                const vector<vector<int>>& coords,
                                size_t order,
                                const size_t fixedLevel,
                                const size_t i, const size_t numCoords) {
  if (i == order) {
    return numCoords;
  }
  if (i == fixedLevel) {
    auto indexValues = getUniqueEntries(coords[i].begin(), coords[i].end());
    return indexValues.size();
  }
  else {
    // Find max occurrences for level i
    size_t maxSize=0;
    vector<int> maxCoords;
    int coordCur=coords[i][0];
    size_t sizeCur=1;
    for (size_t j=1; j<numCoords; j++) {
      if (coords[i][j] == coordCur) {
        sizeCur++;
      }
      else {
        if (sizeCur > maxSize) {
          maxSize = sizeCur;
          maxCoords.clear();
          maxCoords.push_back(coordCur);
        }
        else if (sizeCur == maxSize) {
          maxCoords.push_back(coordCur);
        }
        sizeCur=1;
        coordCur=coords[i][j];
      }
    }
    if (sizeCur > maxSize) {
      maxSize = sizeCur;
      maxCoords.clear();
      maxCoords.push_back(coordCur);
    }
    else if (sizeCur == maxSize) {
      maxCoords.push_back(coordCur);
    }

    size_t maxFixedValue=0;
    size_t maxSegment;
    vector<vector<int>> newCoords(order);
    for (size_t l=0; l<maxCoords.size(); l++) {
      // clean coords for next level
      for (size_t k=0; k<order;k++) {
        newCoords[k].clear();
      }
      for (size_t j=0; j<numCoords; j++) {
        if (coords[i][j] == maxCoords[l]) {
          for (size_t k=0; k<order;k++) {
            newCoords[k].push_back(coords[k][j]);
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

Storage pack(const std::vector<int>&              dimensions,
             const Format&                        format,
             const std::vector<std::vector<int>>& coordinates,
             const std::vector<double>            values) {
  taco_iassert(dimensions.size() == format.getOrder());

  Storage storage(format);

  size_t order = dimensions.size();
  size_t numCoordinates = values.size();

  // Create vectors to store pointers to indices/index sizes
  vector<vector<vector<int>>> indices;
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
        indices[i][0].push_back(0);
        break;
      }
      case Fixed: {
        // Fixed indices have two arrays: a segment array and an index array
        indices.push_back({{}, {}});

        // Add maximum size to segment array
        size_t maxSize = findMaxFixedValue(dimensions, coordinates,
                                           format.getOrder(), i, 0,
                                           numCoordinates);
        taco_iassert(maxSize <= INT_MAX);
        indices[i][0].push_back(static_cast<int>(maxSize));
        break;
      }
    }
  }

  vector<double> vals;
  packTensor(dimensions, coordinates, (const double*)values.data(), 0,
             numCoordinates, format.getModeTypes(), 0, &indices, &vals);

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
        Array pos = makeArray(indices[i][0]);
        Array idx = makeArray(indices[i][1]);
        modeIndices.push_back(ModeIndex({pos, idx}));
        break;
      }
    }
  }
  storage.setIndex(Index(format, modeIndices));
  storage.setValues(makeArray(vals));
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
        : VarAssign::make(Var::make("test", Type::Int), 1.0, true);

    switch (modeType) {
      case Dense: {
        Expr dimension = 10;
        Expr loopVar = Var::make("i", Type::Int);
        insertLoop = ir::For::make(loopVar, 0, dimension, 1, body);
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
