#ifndef TACO_COMPRESSED_MODE_TYPE_H
#define TACO_COMPRESSED_MODE_TYPE_H

#include "taco/lower/mode.h"

namespace taco {

class CompressedModeType : public ModeTypeImpl {
public:
  CompressedModeType();
  CompressedModeType(bool isFull, bool isOrdered,
                     bool isUnique, long long allocSize = 1ll << 20);

  virtual ~CompressedModeType() {}

  virtual ModeType copy(std::vector<ModeType::Property> properties) const;

  virtual ModeFunction posIter(ir::Expr parentPos, Mode mode) const;
  virtual ModeFunction posAccess(ir::Expr parentPos,
                                 std::vector<ir::Expr> coords,
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

  virtual std::vector<ir::Expr> getArrays(ir::Expr tensor, size_t level) const;

protected:
  ir::Expr getPosArray(ModePack pack) const;
  ir::Expr getCoordArray(ModePack pack) const;

  ir::Expr getPosCapacity(Mode mode) const;
  ir::Expr getCoordCapacity(Mode mode) const;

  const long long allocSize;
};

}

#endif
