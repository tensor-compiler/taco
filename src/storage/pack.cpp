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
vector<int> getUniqueEntries(const vector<int>::const_iterator& begin,
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





size_t findMaxFixedValue(const vector<int>& dimensions,
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

ir::Stmt packCode(const Format& format) {
  return ir::Stmt();

#if 0
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
#endif
}

}}
