#ifndef TACO_MODE_FORMAT_SINGLETON_H
#define TACO_MODE_FORMAT_SINGLETON_H

#include "taco/lower/mode_format_impl.h"

namespace taco {

class SingletonModeFormat : public ModeFormatImpl {
public:
  using ModeFormatImpl::getInsertCoord;

  SingletonModeFormat();
  SingletonModeFormat(bool isFull, bool isOrdered, bool isUnique, 
                      bool isZeroless, bool isPadded, 
		      long long allocSize = DEFAULT_ALLOC_SIZE);

  ~SingletonModeFormat() override {}

  ModeFormat copy(std::vector<ModeFormat::Property> properties) const override;

  ModeFunction posIterBounds(ir::Expr parentPos, Mode mode) const override;
  ModeFunction posIterAccess(ir::Expr pos, std::vector<ir::Expr> coords,
                             Mode mode) const override;
  
  ir::Stmt getAppendCoord(ir::Expr pos, ir::Expr coord, 
                          Mode mode) const override; 
  ir::Expr getSize(ir::Expr parentSize, Mode mode) const override;
  ir::Stmt getAppendInitLevel(ir::Expr parentSize, ir::Expr size, 
                              Mode mode) const override;
  ir::Stmt getAppendFinalizeLevel(ir::Expr parentSize, ir::Expr size, 
                                  Mode mode) const override;

  ir::Expr getAssembledSize(ir::Expr prevSize, Mode mode) const override;
  ir::Stmt getInitCoords(ir::Expr prevSize, 
                         std::vector<AttrQueryResult> queries, 
                         Mode mode) const override;
  ModeFunction getYieldPos(ir::Expr parentPos, std::vector<ir::Expr> coords, 
                           Mode mode) const override;
  ir::Stmt getInsertCoord(ir::Expr parentPos, ir::Expr pos, 
                          std::vector<ir::Expr> coords, 
                          Mode mode) const override;

  std::vector<ir::Expr> getArrays(ir::Expr tensor, int mode, 
                                  int level) const override;

protected:
  ir::Expr getCoordArray(ModePack pack) const;

  ir::Expr getCoordCapacity(Mode mode) const;

  bool equals(const ModeFormatImpl& other) const override;

  const long long allocSize;
};

}

#endif
