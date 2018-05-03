#ifndef TACO_ROOT_FORMAT_H
#define TACO_ROOT_FORMAT_H

#include "taco/storage/mode_format.h"

namespace taco {

class RootFormat : public ModeFormat {
public:
  RootFormat();

  virtual ~RootFormat() {}

  virtual ModeType copy(
      const std::vector<ModeType::Property>& properties) const;

  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr> getCoordIter(
      const std::vector<ir::Expr>& i, const ModeType::Mode& mode) const;
  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr> getCoordAccess(
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
};

}

#endif
