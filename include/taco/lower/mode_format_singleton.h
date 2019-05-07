#ifndef TACO_MODE_FORMAT_SINGLETON_H
#define TACO_MODE_FORMAT_SINGLETON_H

#include "taco/lower/mode_format_impl.h"

namespace taco {

class SingletonModeFormat : public ModeFormatImpl {
public:
  SingletonModeFormat();
  SingletonModeFormat(bool isFull, bool isOrdered,
                      bool isUnique, long long allocSize = DEFAULT_ALLOC_SIZE);

  virtual ~SingletonModeFormat() {}

  virtual ModeFormat copy(std::vector<ModeFormat::Property> properties) const;

  virtual ModeFunction posIterBounds(ir::Expr parentPos, Mode mode) const;
  virtual ModeFunction posIterAccess(ir::Expr pos, std::vector<ir::Expr> coords,
                                     Mode mode) const;
  
  virtual ir::Stmt getAppendCoord(ir::Expr pos, ir::Expr coord, 
                                  Mode mode) const; 
  virtual ir::Expr getSize(ir::Expr parentSize, Mode mode) const;
  virtual ir::Stmt getAppendInitLevel(ir::Expr parentSize, ir::Expr size, 
                                      Mode mode) const;
  virtual ir::Stmt getAppendFinalizeLevel(ir::Expr parentSize, ir::Expr size, 
                                          Mode mode) const;

  virtual std::vector<ir::Expr> getArrays(ir::Expr tensor, int mode, 
                                          int level) const;

protected:
  ir::Expr getCoordArray(ModePack pack) const;

  ir::Expr getCoordCapacity(Mode mode) const;

  const long long allocSize;
};

}

#endif
