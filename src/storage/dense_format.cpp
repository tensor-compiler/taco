#include "taco/storage/dense_format.h"

using namespace taco::ir;

namespace taco {

DenseFormat::DenseFormat() : DenseFormat(true, true) {}

DenseFormat::DenseFormat(const bool isOrdered, const bool isUnique) : 
    ModeFormat("dense", true, isOrdered, isUnique, false, true, true, false, 
               true, true, false) {}

ModeType DenseFormat::copy(
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
  return ModeType(std::make_shared<DenseFormat>(isOrdered, isUnique));
}

std::tuple<Stmt,Expr,Expr> DenseFormat::getCoordIter(const std::vector<Expr>& i, 
    const ModeType::Mode& mode) const {
  Expr size = mode.size.isFixed() ? (long long)mode.size.getSize() : 
              getSizeArray(mode.pack);
  return std::tuple<Stmt,Expr,Expr>(Stmt(), 0ll, size);
}

std::tuple<Stmt,Expr,Expr> DenseFormat::getCoordAccess(const Expr& pPrev, 
    const std::vector<Expr>& i, const ModeType::Mode& mode) const {
  Expr size = mode.size.isFixed() ? (long long)mode.size.getSize() : 
              getSizeArray(mode.pack);
  Expr pos = Add::make(Mul::make(pPrev, size), i.back());
  return std::tuple<Stmt,Expr,Expr>(Stmt(), pos, true);
}

std::tuple<Stmt,Expr,Expr> DenseFormat::getLocate(const Expr& pPrev, 
    const std::vector<Expr>& i, const ModeType::Mode& mode) const {
  Expr size = mode.size.isFixed() ? (long long)mode.size.getSize() : 
              getSizeArray(mode.pack);
  Expr pos = Add::make(Mul::make(pPrev, size), i.back());
  return std::tuple<Stmt,Expr,Expr>(Stmt(), pos, true);
}

Stmt DenseFormat::getInsertCoord(const Expr& p, const std::vector<Expr>& i, 
    const ModeType::Mode& mode) const {
  return Stmt();
}

Expr DenseFormat::getSize(const Expr& szPrev, 
    const ModeType::Mode& mode) const {
  Expr size = mode.size.isFixed() ? (long long)mode.size.getSize() : 
              getSizeArray(mode.pack);
  return Mul::make(szPrev, size);
}

Stmt DenseFormat::getInsertInit(const Expr& szPrev, const Expr& sz, 
    const ModeType::Mode& mode) const {
  return Stmt();
}

Stmt DenseFormat::getInsertFinalize(const Expr& szPrev, const Expr& sz, 
    const ModeType::Mode& mode) const {
  return Stmt();
}

Expr DenseFormat::getArray(size_t idx, const ModeType::Mode& mode) const {
  switch (idx) {
    case 0:
      return GetProperty::make(mode.tensor, TensorProperty::Dimension, 
                               mode.mode);
    default:
      break;
  }
  return Expr();
}

Expr DenseFormat::getSizeArray(const ModeType::ModePack* pack) const {
  return pack->getArray(0);
}

}
