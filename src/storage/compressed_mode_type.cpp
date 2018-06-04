#include "taco/storage/compressed_mode_type.h"

using namespace taco::ir;

namespace taco {

CompressedModeType::CompressedModeType() : 
    CompressedModeType(false, true, true) {}

CompressedModeType::CompressedModeType(const bool isFull, const bool isOrdered, 
                                       const bool isUnique, 
                                       const long long allocSize) : 
    ModeTypeImpl("compressed", isFull, isOrdered, isUnique, false, true, false, 
               true, false, false, true), allocSize(allocSize) {}

ModeType CompressedModeType::copy(
    const std::vector<ModeType::Property>& properties) const {
  bool isFull = this->isFull;
  bool isOrdered = this->isOrdered;
  bool isUnique = this->isUnique;
  for (const auto property : properties) {
    switch (property) {
      case ModeType::FULL:
        isFull = true;
        break;
      case ModeType::NOT_FULL:
        isFull = false;
        break;
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
  const auto compressedVariant = 
      std::make_shared<CompressedModeType>(isFull, isOrdered, isUnique);
  return ModeType(compressedVariant);
}

std::tuple<Stmt,Expr,Expr> CompressedModeType::getPosIter(const Expr& pPrev, 
    Mode& mode) const {
  Expr pbegin = Load::make(getPosArray(mode.pack), pPrev);
  Expr pend = Load::make(getPosArray(mode.pack), Add::make(pPrev, 1ll));
  return std::tuple<Stmt,Expr,Expr>(Stmt(), pbegin, pend); 
}

std::tuple<Stmt,Expr,Expr> CompressedModeType::getPosAccess(const Expr& p, 
    const std::vector<Expr>& i, Mode& mode) const {
  Expr idx = Load::make(getIdxArray(mode.pack), p);
  return std::tuple<Stmt,Expr,Expr>(Stmt(), idx, true);
}

Stmt CompressedModeType::getAppendCoord(const ir::Expr& p, const ir::Expr& i, 
    Mode& mode) const {
  Expr idxArray = getIdxArray(mode.pack);
  Stmt storeIdx = Store::make(idxArray, p, i);

  if (mode.pos != (mode.pack->getSize() - 1)) {
    return storeIdx;
  }

  Expr idxCapacity = getIdxCapacity(mode);
  Stmt updateCapacity = VarAssign::make(idxCapacity, Mul::make(2ll, p));
  Stmt resizeIdx = Allocate::make(idxArray, idxCapacity, true);
  Stmt ifBody = Block::make({updateCapacity, resizeIdx});
  Stmt maybeResizeIdx = IfThenElse::make(Lte::make(idxCapacity, p), ifBody);

  return Block::make({maybeResizeIdx, storeIdx});
}

Stmt CompressedModeType::getAppendEdges(const ir::Expr& pPrev, 
    const ir::Expr& pBegin, const ir::Expr& pEnd, Mode& mode) const {
  Expr posArray = getPosArray(mode.pack);
  Expr edges = (!mode.prevModeType.defined() || mode.prevModeType.hasAppend()) ?
      pEnd : Sub::make(pEnd, pBegin);
  return Store::make(posArray, Add::make(pPrev, 1ll), edges);
}

Stmt CompressedModeType::getAppendInitEdges(const ir::Expr& pPrevBegin, 
    const ir::Expr& pPrevEnd, Mode& mode) const {
  if (isa<Literal>(pPrevBegin)) {
    taco_iassert(to<Literal>(pPrevBegin)->equalsScalar(0));
    return Stmt();
  }

  Expr posArray = getPosArray(mode.pack);
  Expr posCapacity = getPosCapacity(mode);
  Expr shouldResize = Lte::make(posCapacity, pPrevEnd);
  Stmt updateCapacity = VarAssign::make(posCapacity, Mul::make(2ll, pPrevEnd));
  Stmt reallocPos = Allocate::make(posArray, posCapacity, true);
  Stmt resizePos = Block::make({updateCapacity, reallocPos});
  Stmt maybeResizePos = IfThenElse::make(shouldResize, resizePos);

  if (!mode.prevModeType.defined() || mode.prevModeType.hasAppend()) {
    return maybeResizePos;
  }

  Expr pVar = Var::make("p" + mode.getName(), Int());
  Expr ub = Add::make(pPrevEnd, 1ll);
  Stmt storePos = Store::make(posArray, pVar, 0ll);
  Stmt initPos = For::make(pVar, pPrevBegin, ub, 1ll, storePos);
  
  return Block::make({maybeResizePos, initPos});
}

Stmt CompressedModeType::getAppendInitLevel(const ir::Expr& szPrev, 
    const ir::Expr& sz, Mode& mode) const {
  Expr posArray = getPosArray(mode.pack);
  Expr posCapacity = getPosCapacity(mode);
  Expr initCapacity = isa<Literal>(szPrev) ? Add::make(szPrev, 1ll) : 
                      Max::make(Add::make(szPrev, 1ll), allocSize);
  Stmt initPosCapacity = VarAssign::make(posCapacity, initCapacity, true);
  Stmt allocPosArray = Allocate::make(posArray, posCapacity);

  Stmt initPos = (!mode.prevModeType.defined() || 
      mode.prevModeType.hasAppend()) ? Store::make(posArray, 0ll, 0ll) : [&]() {
        Expr pVar = Var::make("p" + mode.getName(), Int());
        Stmt storePos = Store::make(posArray, pVar, 0ll);
        return For::make(pVar, 0ll, Add::make(szPrev, 1ll), 1ll, storePos);
      }();
  
  if (mode.pos != (mode.pack->getSize() - 1)) {
    return Block::make({initPosCapacity, allocPosArray, initPos});
  }

  Expr idxCapacity = getIdxCapacity(mode);
  Stmt initIdxCapacity = VarAssign::make(idxCapacity, allocSize, true);
  Stmt allocIdxArray = Allocate::make(getIdxArray(mode.pack), idxCapacity);

  return Block::make({initPosCapacity, initIdxCapacity, allocPosArray, 
                      allocIdxArray, initPos});
}

Stmt CompressedModeType::getAppendFinalizeLevel(const ir::Expr& szPrev, 
    const ir::Expr& sz, Mode& mode) const {
  if ((isa<Literal>(szPrev) && to<Literal>(szPrev)->equalsScalar(1)) || 
      !mode.prevModeType.defined() || mode.prevModeType.hasAppend()) {
    return Stmt();
  }

  Expr csVar = Var::make("cs" + mode.getName(), Int());
  Stmt initCs = VarAssign::make(csVar, 0ll, true);
  
  Expr pVar = Var::make("p" + mode.getName(), Int());
  Expr loadPos = Load::make(getPosArray(mode.pack), pVar);
  Stmt incCs = VarAssign::make(csVar, Add::make(csVar, loadPos));
  Stmt updatePos = Store::make(getPosArray(mode.pack), pVar, csVar);
  Stmt body = Block::make({incCs, updatePos});
  Stmt finalizeLoop = For::make(pVar, 1ll, Add::make(szPrev, 1ll), 1ll, body);

  return Block::make({initCs, finalizeLoop});
}

Expr CompressedModeType::getArray(size_t idx, const Mode& mode) const {
  switch (idx) {
    case 0:
      return GetProperty::make(mode.tensor, TensorProperty::Indices, mode.mode, 
                               0, mode.getName() + "_pos");
    case 1:
      return GetProperty::make(mode.tensor, TensorProperty::Indices, mode.mode, 
                               1, mode.getName() + "_idx");
    default:
      break;
  }
  return Expr();
}

Expr CompressedModeType::getPosArray(const ModePack* pack) const {
  return pack->getArray(0);
}

Expr CompressedModeType::getIdxArray(const ModePack* pack) const {
  return pack->getArray(1);
}

Expr CompressedModeType::getPosCapacity(Mode& mode) const {
  const std::string varName = mode.getName() + "_pos_size";
 
  if (!mode.hasVar(varName)) {
    Expr posCapacity = Var::make(varName, Int());
    mode.addVar(varName, posCapacity);
    return posCapacity;
  }

  return mode.getVar(varName);
}

Expr CompressedModeType::getIdxCapacity(Mode& mode) const {
  const std::string varName = mode.getName() + "_idx_size";
  
  if (!mode.hasVar(varName)) {
    Expr idxCapacity = Var::make(varName, Int());
    mode.addVar(varName, idxCapacity);
    return idxCapacity;
  }

  return mode.getVar(varName);
}

}
