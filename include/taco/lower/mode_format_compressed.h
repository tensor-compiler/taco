#ifndef TACO_MODE_FORMAT_COMPRESSED_H
#define TACO_MODE_FORMAT_COMPRESSED_H

#include "taco/lower/mode_format_impl.h"

namespace taco {

class CompressedModeFormat : public ModeFormatImpl {
public:
  CompressedModeFormat();
  CompressedModeFormat(bool isFull, bool isOrdered,
                       bool isUnique, bool isZeroless, long long allocSize = DEFAULT_ALLOC_SIZE);

  ~CompressedModeFormat() override {}

  ModeFormat copy(std::vector<ModeFormat::Property> properties) const override;
  
  ModeFunction posIterBounds(ir::Expr parentPos, Mode mode) const override;
  ModeFunction posIterAccess(ir::Expr pos, std::vector<ir::Expr> coords,
                                     Mode mode) const override;

  ModeFunction coordBounds(ir::Expr parentPos, Mode mode) const override;
  
  ir::Stmt getAppendCoord(ir::Expr pos, ir::Expr coord, 
                          Mode mode) const override;
  ir::Stmt getAppendEdges(ir::Expr parentPos, ir::Expr posBegin, 
                          ir::Expr posEnd, Mode mode) const override;
  ir::Expr getSize(ir::Expr parentSize, Mode mode) const override;
  ir::Stmt getAppendInitEdges(ir::Expr parentPosBegin, 
                              ir::Expr parentPosEnd, Mode mode) const override;
  ir::Stmt getAppendInitLevel(ir::Expr parentSize, ir::Expr size, 
                              Mode mode) const override;
  ir::Stmt getAppendFinalizeLevel(ir::Expr parentSize, ir::Expr size, 
                                  Mode mode) const override;

  std::vector<ir::Expr> getArrays(ir::Expr tensor, int mode, 
                                  int level) const override;

protected:
  ir::Expr getPosArray(ModePack pack) const;
  ir::Expr getCoordArray(ModePack pack) const;

  ir::Expr getPosCapacity(Mode mode) const;
  ir::Expr getCoordCapacity(Mode mode) const;

  bool equals(const ModeFormatImpl& other) const override;

  const long long allocSize;
};

}

#endif
