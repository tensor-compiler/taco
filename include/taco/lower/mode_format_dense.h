#ifndef TACO_MODE_FORMAT_DENSE_H
#define TACO_MODE_FORMAT_DENSE_H

#include "taco/lower/mode_format_impl.h"

namespace taco {

class DenseModeFormat : public ModeFormatImpl {
public:
  DenseModeFormat();
  DenseModeFormat(const bool isOrdered, const bool isUnique);

  virtual ~DenseModeFormat() {}

  virtual ModeFormat copy(std::vector<ModeFormat::Property> properties) const;
  
  virtual ModeFunction locate(ir::Expr parentPos, std::vector<ir::Expr> coords,
                              Mode mode) const;

  virtual ir::Stmt getInsertCoord(ir::Expr p, const std::vector<ir::Expr>& i, 
                                  Mode mode) const;
  virtual ir::Expr getWidth(Mode mode) const;
  virtual ir::Stmt getInsertInitCoords(ir::Expr pBegin, ir::Expr pEnd, 
                                       Mode mode) const;
  virtual ir::Stmt getInsertInitLevel(ir::Expr szPrev, ir::Expr sz, 
                                      Mode mode) const;
  virtual ir::Stmt getInsertFinalizeLevel(ir::Expr szPrev, ir::Expr sz, 
                                          Mode mode) const;
  
  virtual std::vector<ir::Expr> getArrays(ir::Expr tensor, int mode, 
                                          int level) const;

protected:
  ir::Expr getSizeArray(ModePack pack) const;
};

}

#endif
