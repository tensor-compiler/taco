#include "taco/lower/mode_format_dense_old.h"

using namespace std;
using namespace taco::ir;

namespace taco {
namespace old {

DenseModeFormat::DenseModeFormat() : DenseModeFormat(true, true) {}

DenseModeFormat::DenseModeFormat(const bool isOrdered, const bool isUnique) : 
    ModeFormatImpl("dense", true, isOrdered, isUnique, false, true, true, false, 
                 true, true, false) {}

ModeFormat DenseModeFormat::copy(std::vector<ModeFormat::Property> properties) const {
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
  return ModeFormat(std::make_shared<DenseModeFormat>(isOrdered, isUnique));
}

ModeFunction DenseModeFormat::coordIterBounds(vector<Expr> coords, Mode mode) const {
  return ModeFunction(Stmt(), {0ll, getSize(mode)});
}

ModeFunction DenseModeFormat::coordIterAccess(ir::Expr parentPos,
                                        std::vector<ir::Expr> coords,
                                        Mode mode) const {
  Expr pos = Add::make(Mul::make(parentPos, getSize(mode)), coords.back());
  return ModeFunction(Stmt(), {pos, true});
}

ModeFunction DenseModeFormat::locate(ir::Expr parentPos,
                                   std::vector<ir::Expr> coords,
                                   Mode mode) const {
  Expr pos = Add::make(Mul::make(parentPos, getSize(mode)), coords.back());
  return ModeFunction(Stmt(), {pos, true});
}

Stmt DenseModeFormat::getInsertCoord(Expr p, 
    const std::vector<Expr>& i, Mode mode) const {
  return Stmt();
}

Expr DenseModeFormat::getSize(Mode mode) const {
  return (mode.getSize().isFixed() && mode.getSize().getSize() < 16) ?
         (long long)mode.getSize().getSize() : getSizeArray(mode.getModePack());
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

vector<Expr> DenseModeFormat::getArrays(Expr tensor, int mode) const {
  return {GetProperty::make(tensor, TensorProperty::Dimension, mode-1)};
}

Expr DenseModeFormat::getSizeArray(ModePack pack) const {
  return pack.getArray(0);
}

}}
