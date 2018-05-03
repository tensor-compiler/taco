#include "taco/storage/compressed_format.h"

using namespace taco::ir;

namespace taco {

CompressedFormat::CompressedFormat() : CompressedFormat(false, true, true) {}

CompressedFormat::CompressedFormat(const bool isFull, const bool isOrdered, 
                                   const bool isUnique) : 
    ModeFormat("compressed", isFull, isOrdered, isUnique, false, true, false, 
               true, false, false, true) {}

ModeType CompressedFormat::copy(
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
      std::make_shared<CompressedFormat>(isFull, isOrdered, isUnique);
  return ModeType(compressedVariant);
}

std::tuple<Stmt,Expr,Expr> CompressedFormat::getPosIter(const Expr& pPrev, 
    const ModeType::Mode& mode) const {
  Expr pbegin = Load::make(getPosArray(mode.pack), pPrev);
  Expr pend = Load::make(getPosArray(mode.pack), Add::make(pPrev, 1ll));
  return std::tuple<Stmt,Expr,Expr>(Stmt(), pbegin, pend); 
}

std::tuple<Stmt,Expr,Expr> CompressedFormat::getPosAccess(const Expr& p, 
    const std::vector<Expr>& i, const ModeType::Mode& mode) const {
  Expr idx = Load::make(getIdxArray(mode.pack), p);
  return std::tuple<Stmt,Expr,Expr>(Stmt(), idx, true);
}

Stmt CompressedFormat::getAppendCoord(const Expr& p, const std::vector<Expr>& i, 
    const ModeType::Mode& mode) const {
  return Store::make(getIdxArray(mode.pack), p, i.back());
}

Stmt CompressedFormat::getAppendEdges(const Expr& pPrev, const Expr& pBegin, 
    const Expr& pEnd, const ModeType::Mode& mode) const {
  Expr segSize = Sub::make(pEnd, pBegin);
  return Store::make(getPosArray(mode.pack), Add::make(pPrev, 1ll), segSize);
}

Stmt CompressedFormat::getAppendInit(const Expr& szPrev, const Expr& sz, 
    const ModeType::Mode& mode) const {
  Expr pVar = Var::make("p" + mode.getName(), Int());
  Stmt initPos = Store::make(getPosArray(mode.pack), pVar, 0ll);
  return For::make(pVar, 0ll, Add::make(szPrev, 1ll), 1ll, initPos);
}

Stmt CompressedFormat::getAppendFinalize(const Expr& szPrev, const Expr& sz, 
    const ModeType::Mode& mode) const {
  Expr csVar = Var::make("cs" + mode.getName(), Int());
  Stmt initCs = VarAssign::make(csVar, 0ll, true);
  
  Expr pVar = Var::make("p" + mode.getName(), Int());
  Expr loadPos = Load::make(getPosArray(mode.pack), pVar);
  Stmt incCs = VarAssign::make(csVar, Add::make(csVar, loadPos));
  Stmt updatePos = Store::make(getPosArray(mode.pack), pVar, csVar);
  Stmt loopBody = Block::make({incCs, updatePos});
  Stmt finalizeLoop = For::make(pVar, 1ll, Add::make(szPrev, 1ll), 
                                        1ll, loopBody);

  return Block::make({initCs, finalizeLoop});
}

Expr CompressedFormat::getArray(size_t idx, const ModeType::Mode& mode) const {
  switch (idx) {
    case 0:
      return GetProperty::make(mode.tensor, TensorProperty::Indices, 
                                   mode.mode, 0, mode.getName() + "_pos");
    case 1:
      return GetProperty::make(mode.tensor, TensorProperty::Indices, 
                                   mode.mode, 1, mode.getName() + "_idx");
    default:
      break;
  }
  return Expr();
}

Expr CompressedFormat::getPosArray(const ModeType::ModePack* pack) const {
  return pack->getArray(0);
}

Expr CompressedFormat::getIdxArray(const ModeType::ModePack* pack) const {
  return pack->getArray(1);
}

}
