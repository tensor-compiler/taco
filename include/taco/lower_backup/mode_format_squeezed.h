#ifndef TACO_MODE_FORMAT_SQUEEZED_H
#define TACO_MODE_FORMAT_SQUEEZED_H

#include "taco/lower/mode_format_impl.h"

namespace taco {

class SqueezedModeFormat : public ModeFormatImpl {
public:
  SqueezedModeFormat();
  SqueezedModeFormat(bool isOrdered, bool isUnique);

  virtual ~SqueezedModeFormat() {}

  virtual ModeFormat copy(std::vector<ModeFormat::Property> properties) const;
  
  virtual std::vector<attr_query::AttrQuery>
  attrQueries(std::vector<IndexVarExpr> coords, std::vector<IndexVarExpr> vals) const;
  
  virtual ir::Expr getSizeNew(ir::Expr prevSize, Mode mode) const;
  virtual ir::Stmt getInitCoords(ir::Expr prevSize, 
                                 std::map<std::string,AttrQueryResult> queries, 
                                 Mode mode) const;
  virtual ir::Stmt getInitYieldPos(ir::Expr prevSize, Mode mode) const;
  virtual ModeFunction getYieldPos(ir::Expr parentPos, 
                                   std::vector<ir::Expr> coords, 
                                   Mode mode) const;
  virtual ir::Stmt getFinalizeLevel(ir::Expr prevSize, Mode mode) const;

  virtual std::vector<ir::Expr> getArrays(ir::Expr tensor, int mode, 
                                          int level) const;

protected:
  ir::Expr getPermArray(ModePack pack) const;
  ir::Expr getPermSizeArray(ModePack pack) const;
  ir::Expr getShiftArray(ModePack pack) const;
  ir::Expr getSizeArray(ModePack pack) const;

  ir::Expr getRperm(Mode mode) const;
  ir::Expr getLocalPermSize(Mode mode) const;  // TODO: conversion to local should be handled by lowering machinery
  ir::Expr getLocalShift(Mode mode) const;  // TODO: conversion to local should be handled by lowering machinery
};

}

#endif
