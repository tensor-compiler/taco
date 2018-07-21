#ifndef TACO_COMPRESSED_MODE_TYPE_H
#define TACO_COMPRESSED_MODE_TYPE_H

#include "taco/storage/mode_type.h"

namespace taco {

class CompressedModeType : public ModeTypeImpl {
public:
  CompressedModeType();
  CompressedModeType(bool isFull, bool isOrdered,
                     bool isUnique, long long allocSize = 1ll << 20);

  virtual ~CompressedModeType() {}

  virtual ModeType copy(
      const std::vector<ModeType::Property>& properties) const;
 
  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr> getPosIter(
      ir::Expr pPrev, Mode mode) const;
  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr> getPosAccess(ir::Expr p, 
      const std::vector<ir::Expr>& i, Mode mode) const;
  
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

  virtual ir::Expr getArray(size_t idx, const Mode mode) const;

protected:
  ir::Expr getPosArray(const ModePack* pack) const;
  ir::Expr getIdxArray(const ModePack* pack) const;

  ir::Expr getPosCapacity(Mode mode) const;
  ir::Expr getIdxCapacity(Mode mode) const;

  const long long allocSize;
};

}

#endif
