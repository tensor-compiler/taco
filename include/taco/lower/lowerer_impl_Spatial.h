#ifndef TACO_LOWERER_IMPL_SPATIAL_H
#define TACO_LOWERER_IMPL_SPATIAL_H

#include <vector>
#include <map>
#include <set>
#include <memory>
#include <taco/index_notation/index_notation.h>

#include "taco/lower/iterator.h"
#include "taco/util/scopedset.h"
#include "taco/util/uncopyable.h"
#include "taco/ir_tags.h"

namespace taco {
class LowererImplSpatial : public util::Uncopyable {
public:
  LowererImplSpatial();
  virtual ~LowererImplSpatial() = default;
private:
  class Visitor;
};

} // namespace taco
#endif 
