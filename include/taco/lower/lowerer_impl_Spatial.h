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
#include "taco/lower/lowerer_impl.h"

namespace taco {
class LowererImplSpatial : public LowererImpl {
public:
  LowererImplSpatial();
  virtual ~LowererImplSpatial() = default;

protected:
  /// Retrieve the values array of the tensor var.
  ir::Expr getValuesArray(TensorVar) const;

  /// Initialize temporary variables
  std::vector<ir::Stmt> codeToInitializeTemporary(Where where);

private:
  class Visitor;
  friend class Visitor;
  std::shared_ptr<Visitor> visitor;
};


} // namespace taco
#endif 
