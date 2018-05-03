#ifndef TACO_COMPRESSED_FORMAT_H
#define TACO_COMPRESSED_FORMAT_H

#include "taco/storage/mode_format.h"

namespace taco {

class CompressedFormat : public ModeFormat {
public:
  CompressedFormat();
  CompressedFormat(const bool isFull, const bool isOrdered, 
                   const bool isUnique);

  virtual ~CompressedFormat() {}

  virtual ModeType copy(
      const std::vector<ModeType::Property>& properties) const;
 
  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr> getPosIter(
      const ir::Expr& pPrev, const ModeType::Mode& mode) const;
  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr> getPosAccess(
      const ir::Expr& p, const std::vector<ir::Expr>& i, 
      const ModeType::Mode& mode) const;
  
  virtual ir::Stmt getAppendCoord(const ir::Expr& p, 
      const std::vector<ir::Expr>& i, const ModeType::Mode& mode) const;
  virtual ir::Stmt getAppendEdges(const ir::Expr& pPrev, const ir::Expr& pBegin, 
      const ir::Expr& pEnd, const ModeType::Mode& mode) const;
  virtual ir::Stmt getAppendInit(const ir::Expr& szPrev, const ir::Expr& sz, 
      const ModeType::Mode& mode) const;
  virtual ir::Stmt getAppendFinalize(const ir::Expr& szPrev, const ir::Expr& sz, 
      const ModeType::Mode& mode) const;

  virtual ir::Expr getArray(size_t idx, const ModeType::Mode& mode) const;

protected:
  ir::Expr getPosArray(const ModeType::ModePack* pack) const;
  ir::Expr getIdxArray(const ModeType::ModePack* pack) const;
};

}

#endif
