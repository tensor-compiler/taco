#include "taco/lower/dense_mode_type.h"

using namespace std;
using namespace taco::ir;

namespace taco {

DenseModeType::DenseModeType() : DenseModeType(true, true) {}

DenseModeType::DenseModeType(const bool isOrdered, const bool isUnique) : 
    ModeFormatImpl("dense", true, isOrdered, isUnique, false, true, true, false, 
                 true, true, false) {}

ModeFormat DenseModeType::copy(std::vector<ModeFormat::Property> properties) const {
  bool isOrdered = this->isOrdered;
  bool isUnique = this->isUnique;
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
      default:
        break;
    }
  }
  return ModeFormat(std::make_shared<DenseModeType>(isOrdered, isUnique));
}

ModeFunction DenseModeType::coordIterBounds(vector<Expr> coords, Mode mode) const {
  return ModeFunction(Stmt(), {0ll, getSize(mode)});
}

ModeFunction DenseModeType::coordIterAccess(ir::Expr parentPos,
                                        std::vector<ir::Expr> coords,
                                        Mode mode) const {
  Expr pos = Add::make(Mul::make(parentPos, getSize(mode)), coords.back());
  return ModeFunction(Stmt(), {pos, true});
}

ModeFunction DenseModeType::locate(ir::Expr parentPos,
                                   std::vector<ir::Expr> coords,
                                   Mode mode) const {
  Expr pos = Add::make(Mul::make(parentPos, getSize(mode)), coords.back());
  return ModeFunction(Stmt(), {pos, true});
}

Stmt DenseModeType::getInsertCoord(Expr p, 
    const std::vector<Expr>& i, Mode mode) const {
  return Stmt();
}

Expr DenseModeType::getSize(Mode mode) const {
  return (mode.getSize().isFixed() && mode.getSize().getSize() < 16) ?
         (long long)mode.getSize().getSize() : getSizeArray(mode.getModePack());
}

Stmt DenseModeType::getInsertInitCoords(Expr pBegin, 
    Expr pEnd, Mode mode) const {
  return Stmt();
}

Stmt DenseModeType::getInsertInitLevel(Expr szPrev, Expr sz, 
    Mode mode) const {
  return Stmt();
}

Stmt DenseModeType::getInsertFinalizeLevel(Expr szPrev, 
    Expr sz, Mode mode) const {
  return Stmt();
}

vector<Expr> DenseModeType::getArrays(Expr tensor, size_t level) const {
  return {GetProperty::make(tensor, TensorProperty::Dimension, level-1)};
}

Expr DenseModeType::getSizeArray(ModePack pack) const {
  return pack.getArray(0);
}

}
