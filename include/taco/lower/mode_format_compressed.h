#ifndef TACO_MODE_FORMAT_COMPRESSED_H
#define TACO_MODE_FORMAT_COMPRESSED_H

#include "taco/lower/mode_format_impl.h"

namespace taco {

class CompressedModeFormat : public ModeFormatImpl {
public:
  CompressedModeFormat();
  CompressedModeFormat(bool isFull, bool isOrdered,
                       bool isUnique, long long allocSize = DEFAULT_ALLOC_SIZE);

  virtual ~CompressedModeFormat() {}

  virtual ModeFormat copy(std::vector<ModeFormat::Property> properties) const;

  virtual ModeFunction posIterBounds(ir::Expr parentPos, Mode mode) const;
  virtual ModeFunction posIterAccess(ir::Expr pos, std::vector<ir::Expr> coords,
                                     Mode mode) const;
  
  virtual ir::Stmt getAppendCoord(ir::Expr p, ir::Expr i,
      Mode mode) const; 
  virtual ir::Stmt getAppendEdges(ir::Expr pPrev, ir::Expr pBegin, 
      ir::Expr pEnd, Mode mode) const;
  virtual ir::Stmt getAppendInitEdges(ir::Expr pPrevBegin, 
      ir::Expr pPrevEnd, Mode mode) const;
  virtual ir::Stmt getAppendInitLevel(ir::Expr szPrev, 
      ir::Expr sz, Mode mode) const;
  virtual ir::Stmt getAppendFinalizeLevel(ir::Expr szPrev, 
      ir::Expr sz, Mode mode) const;

  virtual std::vector<ir::Expr> getArrays(ir::Expr tensor, int mode) const;

protected:
  ir::Expr getPosArray(ModePack pack) const;
  ir::Expr getCoordArray(ModePack pack) const;

  ir::Expr getPosCapacity(Mode mode) const;
  ir::Expr getCoordCapacity(Mode mode) const;

  const long long allocSize;
};

}

#endif
