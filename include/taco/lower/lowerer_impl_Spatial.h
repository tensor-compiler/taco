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
  /// Lower an assignment statement.
  virtual ir::Stmt lowerAssignment(Assignment assignment);

  /// Lower an access expression.
  virtual ir::Expr lowerAccess(Access access);

  /// Retrieve the values array of the tensor var.
  ir::Expr getValuesArray(TensorVar) const;

  /// Initialize temporary variables
  std::vector<ir::Stmt> codeToInitializeTemporary(Where where);

  ir::Stmt lowerMergeLattice(MergeLattice lattice, IndexVar coordinateVar,
                                      IndexStmt statement,
                                      const std::set<Access>& reducedAccesses);

  ir::Stmt lowerMergePoint(MergeLattice pointLattice,
                                           ir::Expr coordinate, IndexVar coordinateVar, IndexStmt statement,
                                           const std::set<Access>& reducedAccesses, bool resolvedCoordDeclared);

  ir::Stmt lowerMergeCases(ir::Expr coordinate, IndexVar coordinateVar, IndexStmt stmt,
                                           MergeLattice lattice,
                                           const std::set<Access>& reducedAccesses);

  ir::Stmt lowerForallPosition(Forall forall, Iterator iterator,
                               std::vector<Iterator> locators,
                               std::vector<Iterator> inserters,
                               std::vector<Iterator> appenders,
                               std::set<Access> reducedAccesses,
                               ir::Stmt recoveryStmt);

private:
  class Visitor;
  friend class Visitor;
  std::shared_ptr<Visitor> visitor;

  bool ignoreVectorize = false;

  int markAssignsAtomicDepth = 0;
};


} // namespace taco
#endif 
