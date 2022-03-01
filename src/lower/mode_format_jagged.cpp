#include "taco/lower/mode_format_jagged.h"

#include "taco/util/strings.h"

using namespace std;
using namespace taco::ir;

namespace taco {

JaggedModeFormat::JaggedModeFormat() : JaggedModeFormat(false) {
}

JaggedModeFormat::JaggedModeFormat(const bool isZeroless) :
    ModeFormatImpl("jagged", false, true, true, false, true, isZeroless,
                   true, false, true, false, false, true, false, true) {
}

ModeFormat JaggedModeFormat::copy(
    std::vector<ModeFormat::Property> properties) const {
  bool isZeroless = this->isZeroless;
  for (const auto property : properties) {
    switch (property) {
      case ModeFormat::ZEROLESS:
        isZeroless = true;
        break;
      case ModeFormat::NOT_ZEROLESS:
        isZeroless = false;
        break;	
      default:
        break;
    }
  }
  return ModeFormat(
      std::make_shared<JaggedModeFormat>(isZeroless));
}

ModeFunction JaggedModeFormat::coordIterBounds(Expr parentPos,
                                               std::vector<ir::Expr> coords,
                                               Mode mode) const {
  Expr pBegin = Load::make(getPosArray(mode.getModePack()), parentPos);
  Expr pEnd = Load::make(getPosArray(mode.getModePack()),
                         ir::Add::make(parentPos, 1));
  return ModeFunction(Stmt(), {0, ir::Sub::make(pEnd, pBegin)});
}

ModeFunction JaggedModeFormat::coordIterAccess(ir::Expr parentPos,
                                               std::vector<ir::Expr> coords,
                                               Mode mode) const {
  Expr pBegin = Load::make(getPosArray(mode.getModePack()), parentPos);
  return ModeFunction(Stmt(), {ir::Add::make(pBegin, coords.back()), true});
}

ModeFunction JaggedModeFormat::locate(ir::Expr parentPos,
                                      std::vector<ir::Expr> coords,
                                      Mode mode) const {
  Expr pBegin = Load::make(getPosArray(mode.getModePack()), parentPos);
  Expr pEnd = Load::make(getPosArray(mode.getModePack()),
                         ir::Add::make(parentPos, 1));
  Expr inRange = Lt::make(coords.back(), ir::Sub::make(pEnd, pBegin));
  return ModeFunction(Stmt(), {ir::Add::make(pBegin, coords.back()), inRange});
}

vector<Expr> JaggedModeFormat::getArrays(Expr tensor, int mode, int level) const {
  const auto arraysName = util::toString(tensor) + std::to_string(level);
  return {GetProperty::make(tensor, TensorProperty::Indices,
                            level - 1, 0, arraysName + "_pos")};
}

Expr JaggedModeFormat::getPosArray(ModePack pack) const {
  return pack.getArray(0);
}

}
