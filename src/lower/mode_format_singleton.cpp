#include "taco/lower/mode_format_singleton.h"

#include "ir/ir_generators.h"
#include "taco/ir/simplify.h"
#include "taco/util/strings.h"

using namespace std;
using namespace taco::ir;

namespace taco {

SingletonModeFormat::SingletonModeFormat() : 
    SingletonModeFormat(false, true, true) {
}

SingletonModeFormat::SingletonModeFormat(bool isFull, bool isOrdered,
                                         bool isUnique, long long allocSize) :
    ModeFormatImpl("singleton", isFull, isOrdered, isUnique, true, true,
                   false, true, false, false, true), 
    allocSize(allocSize) {
}

ModeFormat SingletonModeFormat::copy(
    std::vector<ModeFormat::Property> properties) const {
  bool isFull = this->isFull;
  bool isOrdered = this->isOrdered;
  bool isUnique = this->isUnique;
  for (const auto property : properties) {
    switch (property) {
      case ModeFormat::FULL:
        isFull = true;
        break;
      case ModeFormat::NOT_FULL:
        isFull = false;
        break;
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
  const auto singletonVariant = 
      std::make_shared<SingletonModeFormat>(isFull, isOrdered, isUnique);
  return ModeFormat(singletonVariant);
}

ModeFunction SingletonModeFormat::posIterBounds(Expr parentPos, 
                                                Mode mode) const {
  return ModeFunction(Stmt(), {parentPos, Add::make(parentPos, 1)});
}

ModeFunction SingletonModeFormat::posIterAccess(ir::Expr pos,
                                                std::vector<ir::Expr> coords,
                                                Mode mode) const {
  Expr idxArray = getCoordArray(mode.getModePack());
  Expr stride = (int)mode.getModePack().getNumModes();
  Expr offset = (int)mode.getPackLocation();
  Expr loc = Add::make(Mul::make(pos, stride), offset);
  Expr idx = Load::make(idxArray, loc);
  return ModeFunction(Stmt(), {idx, true});
}

Stmt SingletonModeFormat::getAppendCoord(Expr pos, Expr coord, 
                                         Mode mode) const {
  Expr idxArray = getCoordArray(mode.getModePack());
  Expr stride = (int)mode.getModePack().getNumModes();
  Expr offset = (int)mode.getPackLocation();
  Expr loc = Add::make(Mul::make(pos, stride), offset);
  Stmt storeIdx = Store::make(idxArray, loc, coord);

  if (mode.getPackLocation() != (mode.getModePack().getNumModes() - 1)) {
    return storeIdx;
  }

  Expr capacity = getCoordCapacity(mode);
  Stmt maybeResizeIdx = atLeastDoubleSizeIfFull(idxArray, capacity, loc);
  return Block::make(maybeResizeIdx, storeIdx);
}

Expr SingletonModeFormat::getSize(ir::Expr parentSize, Mode mode) const {
  return parentSize;
}

Stmt SingletonModeFormat::getAppendInitLevel(Expr parentSize, Expr size,
                                             Mode mode) const {
  if (mode.getPackLocation() != (mode.getModePack().getNumModes() - 1)) {
    return Stmt();
  }

  Expr defaultCapacity = Literal::make(allocSize, Datatype::Int32); 
  Expr crdCapacity = getCoordCapacity(mode);
  Expr crdArray = getCoordArray(mode.getModePack());
  Stmt initCrdCapacity = VarDecl::make(crdCapacity, defaultCapacity);
  Stmt allocCrd = Allocate::make(crdArray, crdCapacity);

  return Block::make(initCrdCapacity, allocCrd);
}

Stmt SingletonModeFormat::getAppendFinalizeLevel(Expr parentSize, Expr size, 
                                                 Mode mode) const {
  return Stmt();
}

std::vector<Expr> SingletonModeFormat::getArrays(Expr tensor, int mode, 
                                                 int level) const {
  std::string arraysName = util::toString(tensor) + std::to_string(level);
  return {Expr(), 
          GetProperty::make(tensor, TensorProperty::Indices,
                            level - 1, 1, arraysName + "_crd")};
}

Expr SingletonModeFormat::getCoordArray(ModePack pack) const {
  return pack.getArray(1);
}

Expr SingletonModeFormat::getCoordCapacity(Mode mode) const {
  const std::string varName = mode.getName() + "_crd_size";
  
  if (!mode.hasVar(varName)) {
    Expr idxCapacity = Var::make(varName, Int());
    mode.addVar(varName, idxCapacity);
    return idxCapacity;
  }

  return mode.getVar(varName);
}

}
