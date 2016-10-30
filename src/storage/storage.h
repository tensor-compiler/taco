#ifndef TACO_PACKED_TENSOR_H
#define TACO_PACKED_TENSOR_H

#include <cstdlib>
#include <utility>
#include <vector>
#include <inttypes.h>
#include <ostream>
#include <iostream>
#include <string.h>
#include <memory>

#include "format.h"
#include "util/collections.h"
#include "util/strings.h"
#include "util/uncopyable.h"

namespace taco {
namespace storage {

/// The index storage for one tree level.
class LevelStorage {
public:
  LevelStorage(LevelType levelType);

  /// Set the level storage's ptr. This pointer will be freed by level storage.
  void setPtr(int* ptr);

  /// Set the level storage's idx. This pointer will be freed by level storage.
  void setIdx(int* idx);

  void setPtr(const std::vector<int>& ptrVector);
  void setIdx(const std::vector<int>& idxVector);

  // TODO: Remove these functions
  std::vector<int> getPtrAsVector() const;
  std::vector<int> getIdxAsVector() const;

  int getPtrSize() const;
  int getIdxSize() const;

  const int* getPtr() const;
  const int* getIdx() const;

  int* getPtr();
  int* getIdx();

private:
  struct Content;
  std::shared_ptr<Content> content;
};

class Storage {
public:
  struct Size {
    struct LevelIndexSize {
      size_t ptr;
      size_t idx;
    };
    std::vector<LevelIndexSize> levelIndices;
    size_t values;
  };

  struct LevelIndex {
    int* ptr;
    int* idx;
  };

  Storage();
  Storage(const Format& format);

  void setLevelIndex(int level, int* ptr, int* idx);
  void setValues(double* vals);

  const Format& getFormat() const;

  const Storage::LevelIndex& getLevelIndex(int level) const;
  Storage::LevelIndex& getLevelIndex(int level);

  const double* getValues() const;
  double*& getValues();

  /** Returns the size of the idx/ptr arrays of each index.
    * Note that the sizes are computed on demand and that the cost of this
    * function is O(#level).
    */
  Storage::Size getSize() const;

  bool defined() const;

private:
  struct Content;
  std::shared_ptr<Content> content;

  std::vector<LevelStorage> levelStorage;
};

std::ostream& operator<<(std::ostream&, const Storage&);

}}
#endif
