#ifndef TACO_MODE_FORMAT_OFFSET_H
#define TACO_MODE_FORMAT_OFFSET_H

#include "taco/lower/mode_format_impl.h"

namespace taco {

class OffsetModeFormat : public ModeFormatImpl {
public:
  OffsetModeFormat();
  OffsetModeFormat(const bool isOrdered, const bool isUnique);

  virtual ~OffsetModeFormat() {}

  virtual ModeFormat copy(std::vector<ModeFormat::Property> properties) const;
  
  virtual ir::Expr getSizeNew(ir::Expr prevSize, Mode mode) const;
  virtual ModeFunction getYieldPos(ir::Expr parentPos, 
                                   std::vector<ir::Expr> coords, 
                                   Mode mode) const;
  
  virtual std::vector<ir::Expr> getArrays(ir::Expr tensor, int mode, 
                                          int level) const;

protected:
  //ir::Expr getSizeArray(ModePack pack) const;
};

}

#endif
