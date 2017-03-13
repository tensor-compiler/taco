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

#include "taco/format.h"
#include "util/collections.h"
#include "util/strings.h"
#include "util/uncopyable.h"

namespace taco {
namespace storage {

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

  void setLevelIndex(size_t level, int* ptr, int* idx);
  void setValues(double* vals);

  const Format& getFormat() const;

  const Storage::LevelIndex& getLevelIndex(size_t level) const;
  Storage::LevelIndex& getLevelIndex(size_t level);

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
};

std::ostream& operator<<(std::ostream&, const Storage&);

}}
#endif
