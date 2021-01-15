#ifndef TACO_MODE_FORMAT_BLOCK_H
#define TACO_MODE_FORMAT_BLOCK_H

#include "taco/lower/mode_format_impl.h"

namespace taco {

class BlockModeFormat : public ModeFormatImpl {
public:
  BlockModeFormat(int size);
  BlockModeFormat(int size, const bool isOrdered, const bool isUnique);

  virtual ~BlockModeFormat() {}

  virtual ModeFormat copy(std::vector<ModeFormat::Property> properties) const;
  
  virtual ModeFunction locate(ir::Expr parentPos, std::vector<ir::Expr> coords,
                              Mode mode) const;

  virtual ir::Expr getWidth(Mode mode) const;
  
  virtual ir::Expr getSizeNew(ir::Expr prevSize, Mode mode) const;
  virtual ModeFunction getYieldPos(ir::Expr parentPos, 
                                   std::vector<ir::Expr> coords, 
                                   Mode mode) const;
  
  virtual std::vector<ir::Expr> getArrays(ir::Expr tensor, int mode, 
                                          int level) const;

protected:
  ir::Expr getSizeArray(ModePack pack) const;

  const int size;
};

}

#endif
