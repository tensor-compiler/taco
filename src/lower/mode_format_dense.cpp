#include "taco/lower/mode_format_dense.h"

using namespace std;
using namespace taco::ir;

namespace taco {

DenseModeFormat::DenseModeFormat() : DenseModeFormat(true, true, false) {
}

DenseModeFormat::DenseModeFormat(const bool isOrdered, const bool isUnique, 
                                 const bool isZeroless) : 
    ModeFormatImpl("dense", true, isOrdered, isUnique, false, true, isZeroless, 
                   true, false, false, true, true, false, false, false, true) {
}

ModeFormat DenseModeFormat::copy(
    std::vector<ModeFormat::Property> properties) const {
  bool isOrdered = this->isOrdered;
  bool isUnique = this->isUnique;
  bool isZeroless = this->isZeroless;  
  for (const auto property : properties) {
    switch (property) {
      case ModeFormat::ORDERED:
        isOrdered = true;
        break;
      case ModeFormat::NOT_ORDERED:
        isOrdered = false;
        break;
      case ModeFormat::UNIQUE:
        isUnique = true;
        break;
      case ModeFormat::NOT_UNIQUE:
        isUnique = false;
        break;
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
      std::make_shared<DenseModeFormat>(isOrdered, isUnique, isZeroless));  
}

ModeFunction DenseModeFormat::locate(ir::Expr parentPos,
                                   std::vector<ir::Expr> coords,
                                   Mode mode) const {
  Expr pos = ir::Add::make(ir::Mul::make(parentPos, getWidth(mode)), coords.back());
  return ModeFunction(Stmt(), {pos, true});
}

Stmt DenseModeFormat::getInsertCoord(Expr p, 
    const std::vector<Expr>& i, Mode mode) const {
  return Stmt();
}

Expr DenseModeFormat::getWidth(Mode mode) const {
  return (mode.getSize().isFixed() && mode.getSize().getSize() < 16) ?
         (int)mode.getSize().getSize() : 
         getSizeArray(mode.getModePack());
}

Stmt DenseModeFormat::getInsertInitCoords(Expr pBegin, 
    Expr pEnd, Mode mode) const {
  return Stmt();
}

Stmt DenseModeFormat::getInsertInitLevel(Expr szPrev, Expr sz, 
    Mode mode) const {
  return Stmt();
}

Stmt DenseModeFormat::getInsertFinalizeLevel(Expr szPrev, 
    Expr sz, Mode mode) const {
  return Stmt();
}

Expr DenseModeFormat::getAssembledSize(Expr prevSize, Mode mode) const {
  return ir::Mul::make(prevSize, getWidth(mode));
}

ModeFunction DenseModeFormat::getYieldPos(Expr parentPos, 
    std::vector<Expr> coords, Mode mode) const {
  return locate(parentPos, coords, mode);
}

vector<Expr> DenseModeFormat::getArrays(Expr tensor, int mode, 
                                        int level) const {
  return {GetProperty::make(tensor, TensorProperty::Dimension, mode)};
}

Expr DenseModeFormat::getSizeArray(ModePack pack) const {
  return pack.getArray(0);
}

}
