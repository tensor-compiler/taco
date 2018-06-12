#include "taco/storage/dense_mode_type.h"

using namespace taco::ir;

namespace taco {

DenseModeType::DenseModeType() : DenseModeType(true, true) {}

DenseModeType::DenseModeType(const bool isOrdered, const bool isUnique) : 
    ModeTypeImpl("dense", true, isOrdered, isUnique, false, true, true, false, 
                 true, true, false) {}

ModeType DenseModeType::copy(
    const std::vector<ModeType::Property>& properties) const {
  bool isOrdered = this->isOrdered;
  bool isUnique = this->isUnique;
  for (const auto property : properties) {
    switch (property) {
      case ModeType::ORDERED:
        isOrdered = true;
        break;
      case ModeType::NOT_ORDERED:
        isOrdered = false;
        break;
      case ModeType::UNIQUE:
        isUnique = true;
        break;
      case ModeType::NOT_UNIQUE:
        isUnique = false;
        break;
      default:
        break;
    }
  }
  return ModeType(std::make_shared<DenseModeType>(isOrdered, isUnique));
}

std::tuple<Stmt,Expr,Expr> DenseModeType::getCoordIter(const std::vector<Expr>& i, 
    Mode& mode) const {
  return std::tuple<Stmt,Expr,Expr>(Stmt(), 0ll, getSize(mode));
}

std::tuple<Stmt,Expr,Expr> DenseModeType::getCoordAccess(const Expr& pPrev, 
    const std::vector<Expr>& i, Mode& mode) const {
  Expr pos = Add::make(Mul::make(pPrev, getSize(mode)), i.back());
  return std::tuple<Stmt,Expr,Expr>(Stmt(), pos, true);
}

std::tuple<Stmt,Expr,Expr> DenseModeType::getLocate(const Expr& pPrev, 
    const std::vector<Expr>& i, Mode& mode) const {
  Expr pos = Add::make(Mul::make(pPrev, getSize(mode)), i.back());
  return std::tuple<Stmt,Expr,Expr>(Stmt(), pos, true);
}

Stmt DenseModeType::getInsertCoord(const ir::Expr& p, 
    const std::vector<ir::Expr>& i, Mode& mode) const {
  return Stmt();
}

Expr DenseModeType::getSize(Mode& mode) const {
  return (mode.size.isFixed() && mode.size.getSize() < 16) ? 
         (long long)mode.size.getSize() : getSizeArray(mode.pack);
}

Stmt DenseModeType::getInsertInitCoords(const ir::Expr& pBegin, 
    const ir::Expr& pEnd, Mode& mode) const {
  return Stmt();
}

Stmt DenseModeType::getInsertInitLevel(const ir::Expr& szPrev, const ir::Expr& sz, 
    Mode& mode) const {
  return Stmt();
}

Stmt DenseModeType::getInsertFinalizeLevel(const ir::Expr& szPrev, 
    const ir::Expr& sz, Mode& mode) const {
  return Stmt();
}

Expr DenseModeType::getArray(size_t idx, const Mode& mode) const {
  switch (idx) {
    case 0:
      return GetProperty::make(mode.tensor, TensorProperty::Dimension, 
                               mode.mode);
    default:
      break;
  }
  return Expr();
}

Expr DenseModeType::getSizeArray(const ModePack* pack) const {
  return pack->getArray(0);
}

}
