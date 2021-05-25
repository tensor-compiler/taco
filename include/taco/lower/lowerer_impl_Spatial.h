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
#include "taco/lower/lowerer_impl_dataflow.h"

namespace taco {
class LowererImplSpatial : public LowererImplDataflow {
public:
  LowererImplSpatial();
  virtual ~LowererImplSpatial() = default;

protected:
  /// Lower an assignment statement.
  ir::Stmt lowerAssignment(Assignment assignment) override;

  /// Lower an access expression.
  ir::Expr lowerAccess(Access access) override;

  /// Retrieve the values array of the tensor var.
  ir::Expr getValuesArray(TensorVar) const override;

  /// Initialize temporary variables
  std::vector<ir::Stmt> codeToInitializeTemporary(Where where) override;

  ir::Stmt lowerMergeLattice(MergeLattice lattice, IndexVar coordinateVar,
                                      IndexStmt statement,
                                      const std::set<Access>& reducedAccesses) override;

  ir::Stmt lowerMergePoint(MergeLattice pointLattice,
                                           ir::Expr coordinate, IndexVar coordinateVar, IndexStmt statement,
                                           const std::set<Access>& reducedAccesses, bool resolvedCoordDeclared) override;

  ir::Stmt lowerMergeCases(ir::Expr coordinate, IndexVar coordinateVar, IndexStmt stmt,
                                           MergeLattice lattice,
                                           const std::set<Access>& reducedAccesses) override;

  ir::Stmt lowerForallDimension(Forall forall,
                                        std::vector<Iterator> locaters,
                                        std::vector<Iterator> inserters,
                                        std::vector<Iterator> appenders,
                                        std::set<Access> reducedAccesses,
                                        ir::Stmt recoveryStmt) override;

  ir::Stmt lowerForallPosition(Forall forall, Iterator iterator,
                               std::vector<Iterator> locators,
                               std::vector<Iterator> inserters,
                               std::vector<Iterator> appenders,
                               std::set<Access> reducedAccesses,
                               ir::Stmt recoveryStmt) override;

private:
  class Visitor;
  friend class Visitor;
  std::shared_ptr<Visitor> visitor;

  bool ignoreVectorize = false;

  int markAssignsAtomicDepth = 0;
};


} // namespace taco
#endif 
