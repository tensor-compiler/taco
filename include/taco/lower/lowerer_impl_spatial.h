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
namespace ir {
class LowererImplSpatial : public LowererImpl {
public:
  LowererImplSpatial();
  virtual ~LowererImplSpatial() = default;

  /// Lower an index statement to an IR function for 
  /// Spatial code generation. This should override LowererImpl's ir::visit
  ir::Stmt lower(IndexStmt stmt, std::string name, 
                 bool assemble, bool compute, bool pack, bool unpack);
protected:

}

} // namespace ir
} // namespace taco
