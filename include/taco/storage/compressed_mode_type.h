#ifndef TACO_COMPRESSED_MODE_TYPE_H
#define TACO_COMPRESSED_MODE_TYPE_H

#include "taco/storage/mode_type.h"

namespace taco {

class CompressedModeType : public ModeTypeImpl {
public:
  CompressedModeType();
  CompressedModeType(const bool isFull, const bool isOrdered, 
                     const bool isUnique, 
                     const long long allocSize = 1ll << 20);

  virtual ~CompressedModeType() {}

  virtual ModeType copy(
      const std::vector<ModeType::Property>& properties) const;
 
  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr> getPosIter(
      const ir::Expr& pPrev, Mode& mode) const;
  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr> getPosAccess(const ir::Expr& p, 
      const std::vector<ir::Expr>& i, Mode& mode) const;
  
  virtual ir::Stmt getAppendCoord(const ir::Expr& p, const ir::Expr& i, 
      Mode& mode) const; 
  virtual ir::Stmt getAppendEdges(const ir::Expr& pPrev, const ir::Expr& pBegin, 
      const ir::Expr& pEnd, Mode& mode) const;
  virtual ir::Stmt getAppendInitEdges(const ir::Expr& pPrevBegin, 
      const ir::Expr& pPrevEnd, Mode& mode) const;
  virtual ir::Stmt getAppendInitLevel(const ir::Expr& szPrev, 
      const ir::Expr& sz, Mode& mode) const;
  virtual ir::Stmt getAppendFinalizeLevel(const ir::Expr& szPrev, 
      const ir::Expr& sz, Mode& mode) const;

  virtual ir::Expr getArray(size_t idx, const Mode& mode) const;

protected:
  ir::Expr getPosArray(const ModePack* pack) const;
  ir::Expr getIdxArray(const ModePack* pack) const;

  ir::Expr getPosCapacity(Mode& mode) const;
  ir::Expr getIdxCapacity(Mode& mode) const;

  const long long allocSize;
};

}

#endif
