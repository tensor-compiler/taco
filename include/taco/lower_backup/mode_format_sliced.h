#ifndef TACO_MODE_FORMAT_SLICED_H
#define TACO_MODE_FORMAT_SLICED_H

#include "taco/lower/mode_format_impl.h"

namespace taco {

class SlicedModeFormat : public ModeFormatImpl {
public:
  SlicedModeFormat();
  SlicedModeFormat(const bool isOrdered, const bool isUnique);

  virtual ~SlicedModeFormat() {}

  virtual ModeFormat copy(std::vector<ModeFormat::Property> properties) const;
  
  virtual std::vector<attr_query::AttrQuery>
  attrQueries(std::vector<IndexVarExpr> coords, std::vector<IndexVarExpr> vals) const;
  
  virtual ModeFunction locate(ir::Expr parentPos, std::vector<ir::Expr> coords,
                              Mode mode) const;

  virtual ir::Stmt getInsertCoord(ir::Expr p, const std::vector<ir::Expr>& i, 
                                  Mode mode) const;
  virtual ir::Expr getWidth(Mode mode) const;
  virtual ir::Stmt getInsertInitCoords(ir::Expr pBegin, ir::Expr pEnd, 
                                       Mode mode) const;
  virtual ir::Stmt getInsertInitLevel(ir::Expr szPrev, ir::Expr sz, 
                                      Mode mode) const;
  virtual ir::Stmt getInsertFinalizeLevel(ir::Expr szPrev, ir::Expr sz, 
                                          Mode mode) const;
  
  virtual ir::Expr getSizeNew(ir::Expr prevSize, Mode mode) const;
  virtual ir::Stmt getInitCoords(ir::Expr prevSize, 
                                 std::map<std::string,AttrQueryResult> queries, 
                                 Mode mode) const;
  virtual ModeFunction getYieldPos(ir::Expr parentPos, 
                                   std::vector<ir::Expr> coords, 
                                   Mode mode) const;
  
  virtual std::vector<ir::Expr> getArrays(ir::Expr tensor, int mode, 
                                          int level) const;

protected:
  ir::Expr getUBArray(ModePack pack) const;
};

}

#endif
