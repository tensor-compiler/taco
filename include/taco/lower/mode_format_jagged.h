#ifndef TACO_MODE_FORMAT_JAGGED_H
#define TACO_MODE_FORMAT_JAGGED_H

#include "taco/lower/mode_format_impl.h"

namespace taco {

class JaggedModeFormat : public ModeFormatImpl {
public:
  JaggedModeFormat();
  JaggedModeFormat(bool isZeroless);

  ~JaggedModeFormat() override {}

  ModeFormat copy(std::vector<ModeFormat::Property> properties) const override;
  
  //std::vector<AttrQuery>
  //attrQueries(std::vector<IndexVar> parentCoords,
  //            std::vector<IndexVar> childCoords) const override;

  ModeFunction coordIterBounds(ir::Expr parentPos, std::vector<ir::Expr> coords,
                               Mode mode) const override;
  ModeFunction coordIterAccess(ir::Expr parentPos, std::vector<ir::Expr> coords,
                               Mode mode) const override;

  ModeFunction locate(ir::Expr parentPos, std::vector<ir::Expr> coords,
                      Mode mode) const override;

  //ir::Expr getAssembledSize(ir::Expr prevSize, Mode mode) const override;
  //ir::Stmt getSeqInitEdges(ir::Expr prevSize,
  //                         std::vector<AttrQueryResult> queries,
  //                         Mode mode) const override;
  //ir::Stmt getSeqInsertEdge(ir::Expr parentPos,
  //                          std::vector<ir::Expr> coords,
  //                          std::vector<AttrQueryResult> queries,
  //                          Mode mode) const override;
  //ir::Stmt getInitCoords(ir::Expr prevSize,
  //                       std::vector<AttrQueryResult> queries,
  //                       Mode mode) const override;
  //ModeFunction getYieldPos(ir::Expr parentPos, std::vector<ir::Expr> coords,
  //                         Mode mode) const override;

  std::vector<ir::Expr> getArrays(ir::Expr tensor, int mode,
                                  int level) const override;

protected:
  ir::Expr getPosArray(ModePack pack) const;
};

}

#endif
