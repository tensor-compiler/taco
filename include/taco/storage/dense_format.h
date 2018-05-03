#ifndef TACO_DENSE_FORMAT_H
#define TACO_DENSE_FORMAT_H

#include "taco/storage/mode_format.h"

namespace taco {

class DenseFormat : public ModeFormat {
public:
  DenseFormat();
  DenseFormat(const bool isOrdered, const bool isUnique);

  virtual ~DenseFormat() {}

  virtual ModeType copy(
      const std::vector<ModeType::Property>& properties) const;

  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr> getCoordIter(
      const std::vector<ir::Expr>& i, const ModeType::Mode& mode) const;
  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr> getCoordAccess(
      const ir::Expr& pPrev, const std::vector<ir::Expr>& i, 
      const ModeType::Mode& mode) const;
  
  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr> getLocate(
      const ir::Expr& pPrev, const std::vector<ir::Expr>& i, 
      const ModeType::Mode& mode) const;

  virtual ir::Stmt getInsertCoord(const ir::Expr& p, 
      const std::vector<ir::Expr>& i, const ModeType::Mode& mode) const;
  virtual ir::Expr getSize(const ir::Expr& szPrev, 
      const ModeType::Mode& mode) const;
  virtual ir::Stmt getInsertInit(const ir::Expr& szPrev, const ir::Expr& sz, 
      const ModeType::Mode& mode) const;
  virtual ir::Stmt getInsertFinalize(const ir::Expr& szPrev, const ir::Expr& sz, 
      const ModeType::Mode& mode) const;
  
  virtual ir::Expr getArray(size_t idx, const ModeType::Mode& mode) const;

protected:
  ir::Expr getSizeArray(const ModeType::ModePack* pack) const;
};

}

#endif
