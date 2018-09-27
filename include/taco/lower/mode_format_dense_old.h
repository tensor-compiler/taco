#ifndef TACO_MODE_FORMAT_DENSE_OLD_H
#define TACO_MODE_FORMAT_DENSE_OLD_H

#include "taco/lower/mode_format_impl.h"

namespace taco {
namespace old {

class DenseModeFormat : public ModeFormatImpl {
public:
  DenseModeFormat();
  DenseModeFormat(const bool isOrdered, const bool isUnique);

  virtual ~DenseModeFormat() {}

  virtual ModeFormat copy(std::vector<ModeFormat::Property> properties) const;

  virtual ModeFunction coordIterBounds(std::vector<ir::Expr> parentCoords,
                                   Mode mode) const;
  virtual ModeFunction coordIterAccess(ir::Expr parentPos,
                                   std::vector<ir::Expr> coords,
                                   Mode mode) const;

  virtual ModeFunction locate(ir::Expr parentPos,
                              std::vector<ir::Expr> coords,
                              Mode mode) const;

  virtual ir::Stmt getInsertCoord(ir::Expr p,
      const std::vector<ir::Expr>& i, Mode mode) const;
  virtual ir::Expr getSize(Mode mode) const;
  virtual ir::Stmt getInsertInitCoords(ir::Expr pBegin,
      ir::Expr pEnd, Mode mode) const;
  virtual ir::Stmt getInsertInitLevel(ir::Expr szPrev,
      ir::Expr sz, Mode mode) const;
  virtual ir::Stmt getInsertFinalizeLevel(ir::Expr szPrev,
      ir::Expr sz, Mode mode) const;
  
  virtual std::vector<ir::Expr> getArrays(ir::Expr tensor, int mode) const;

protected:
  ir::Expr getSizeArray(ModePack pack) const;
};

}}

#endif
