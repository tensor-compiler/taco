#ifndef TACO_DENSE_MODE_TYPE_H
#define TACO_DENSE_MODE_TYPE_H

#include "taco/storage/mode_type.h"

namespace taco {

class DenseModeType : public ModeTypeImpl {
public:
  DenseModeType();
  DenseModeType(const bool isOrdered, const bool isUnique);

  virtual ~DenseModeType() {}

  virtual ModeType copy(
      const std::vector<ModeType::Property>& properties) const;

  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr> getCoordIter(
      const std::vector<ir::Expr>& i, Mode& mode) const;
  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr> getCoordAccess(
      const ir::Expr& pPrev, const std::vector<ir::Expr>& i, Mode& mode) const;
  
  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr> getLocate(
      const ir::Expr& pPrev, const std::vector<ir::Expr>& i, Mode& mode) const;

  virtual ir::Stmt getInsertCoord(const ir::Expr& p, 
      const std::vector<ir::Expr>& i, Mode& mode) const;
  virtual ir::Expr getSize(Mode& mode) const;
  virtual ir::Stmt getInsertInitCoords(const ir::Expr& pBegin, 
      const ir::Expr& pEnd, Mode& mode) const;
  virtual ir::Stmt getInsertInitLevel(const ir::Expr& szPrev, 
      const ir::Expr& sz, Mode& mode) const;
  virtual ir::Stmt getInsertFinalizeLevel(const ir::Expr& szPrev, 
      const ir::Expr& sz, Mode& mode) const;
  
  
  virtual ir::Expr getArray(size_t idx, const Mode& mode) const;

protected:
  ir::Expr getSizeArray(const ModePack* pack) const;
};

}

#endif
