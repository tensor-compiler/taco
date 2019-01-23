#include "taco/lower/mode_format_compressed.h"

#include "ir/ir_generators.h"
#include "taco/ir/simplify.h"
#include "taco/util/strings.h"

using namespace std;
using namespace taco::ir;

namespace taco {

CompressedModeFormat::CompressedModeFormat() : 
    CompressedModeFormat(false, true, true) {}

CompressedModeFormat::CompressedModeFormat(bool isFull, bool isOrdered,
                                       bool isUnique, long long allocSize) :
    ModeFormatImpl("compressed", isFull, isOrdered, isUnique, false, true, false, 
               true, false, false, true), allocSize(allocSize) {}

ModeFormat CompressedModeFormat::copy(vector<ModeFormat::Property> properties) const {
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
  const auto compressedVariant = 
      std::make_shared<CompressedModeFormat>(isFull, isOrdered, isUnique);
  return ModeFormat(compressedVariant);
}

ModeFunction CompressedModeFormat::posIterBounds(Expr parentPos, Mode mode) const {
  Expr pbegin = Load::make(getPosArray(mode.getModePack()), parentPos);
  Expr pend = Load::make(getPosArray(mode.getModePack()),
                         Add::make(parentPos, 1));
  return ModeFunction(Stmt(), {pbegin, pend});
}

ModeFunction CompressedModeFormat::posIterAccess(ir::Expr parentPos,
                                                 std::vector<ir::Expr> coords,
                                                 Mode mode) const {
  Expr idx = Load::make(getCoordArray(mode.getModePack()), parentPos);
  return ModeFunction(Stmt(), {idx, true});
}

Stmt CompressedModeFormat::getAppendCoord(Expr p, Expr i, 
    Mode mode) const {
  Expr idxArray = getCoordArray(mode.getModePack());
  Stmt storeIdx = Store::make(idxArray, p, i);

  if (mode.getPackLocation() != (mode.getModePack().getNumModes() - 1)) {
    return storeIdx;
  }

  Stmt maybeResizeIdx = doubleSizeIfFull(idxArray, getCoordCapacity(mode), p);
  return Block::make({maybeResizeIdx, storeIdx});
}

Stmt CompressedModeFormat::getAppendEdges(Expr pPrev, 
    Expr pBegin, Expr pEnd, Mode mode) const {
  Expr posArray = getPosArray(mode.getModePack());
  ModeFormat parentModeType = mode.getParentModeType();
  Expr edges = (!parentModeType.defined() || parentModeType.hasAppend())
               ? pEnd : Sub::make(pEnd, pBegin);
  return Store::make(posArray, Add::make(pPrev, 1), edges);
}

Stmt CompressedModeFormat::getAppendInitEdges(Expr pPrevBegin, 
    Expr pPrevEnd, Mode mode) const {
  if (isa<Literal>(pPrevBegin)) {
    taco_iassert(to<Literal>(pPrevBegin)->equalsScalar(0));
    return Stmt();
  }

  Expr posArray = getPosArray(mode.getModePack());
  Stmt maybeResizePos = doubleSizeIfFull(posArray, getPosCapacity(mode),
                                         pPrevEnd);

  ModeFormat parentModeType = mode.getParentModeType();
  if (!parentModeType.defined() || parentModeType.hasAppend()) {
    return maybeResizePos;
  }

  Expr pVar = Var::make("p" + mode.getName(), Int());
  Expr ub = Add::make(pPrevEnd, 1);
  Stmt storePos = Store::make(posArray, pVar, 0);
  Stmt initPos = For::make(pVar, pPrevBegin, ub, 1, storePos);
  
  return Block::make({maybeResizePos, initPos});
}

Stmt CompressedModeFormat::getAppendInitLevel(Expr szPrev, Expr sz,
                                              Mode mode) const {
  Expr capacity = Literal::make(allocSize, Datatype::Int32);
  Expr posArray = getPosArray(mode.getModePack());
  Expr posCapacity = getPosCapacity(mode);
  Expr initCapacity = isa<Literal>(szPrev)
                      ? Add::make(szPrev, 1)
                      : Max::make(Add::make(szPrev, 1), capacity);
  Stmt initPosCapacity = VarDecl::make(posCapacity, initCapacity);
  Stmt allocPosArray = Allocate::make(posArray, posCapacity);

  Stmt initPos =
      (!mode.getParentModeType().defined() ||
        mode.getParentModeType().hasAppend())
      ? Store::make(posArray, 0, 0)
      : [&]() {
          Expr pVar = Var::make("p" + mode.getName(), Int());
          Stmt storePos = Store::make(posArray, pVar, 0);
          return For::make(pVar, 0, Add::make(szPrev,1), 1, storePos);
        }();
  
  if (mode.getPackLocation() != (mode.getModePack().getNumModes()-1)) {
    return Block::make({initPosCapacity, allocPosArray, initPos});
  }

  Expr idxCapacity = getCoordCapacity(mode);
  Stmt initIdxCapacity = VarDecl::make(idxCapacity, capacity);
  Stmt allocIdxArray = Allocate::make(getCoordArray(mode.getModePack()),
                                      idxCapacity);
  return Block::make({initPosCapacity,
                      allocPosArray,
                      initPos,
                      initIdxCapacity,
                      allocIdxArray});
}

Stmt CompressedModeFormat::getAppendFinalizeLevel(Expr szPrev, 
    Expr sz, Mode mode) const {
    ModeFormat parentModeType = mode.getParentModeType();
  if ((isa<Literal>(szPrev) && to<Literal>(szPrev)->equalsScalar(1)) || 
      !parentModeType.defined() || parentModeType.hasAppend()) {
    return Stmt();
  }

  Expr csVar = Var::make("cs" + mode.getName(), Int());
  Stmt initCs = VarDecl::make(csVar, 0);
  
  Expr pVar = Var::make("p" + mode.getName(), Int());
  Expr loadPos = Load::make(getPosArray(mode.getModePack()), pVar);
  Stmt incCs = Assign::make(csVar, Add::make(csVar, loadPos));
  Stmt updatePos = Store::make(getPosArray(mode.getModePack()), pVar, csVar);
  Stmt body = Block::make({incCs, updatePos});
  Stmt finalizeLoop = For::make(pVar, 1, Add::make(szPrev, 1), 1, body);

  return Block::make({initCs, finalizeLoop});
}

vector<Expr> CompressedModeFormat::getArrays(Expr tensor, int mode) const {
  std::string arraysName = util::toString(tensor) + std::to_string(mode);
  return {GetProperty::make(tensor, TensorProperty::Indices,
                            mode-1, 0, arraysName+"_pos"),
          GetProperty::make(tensor, TensorProperty::Indices,
                            mode-1, 1, arraysName+"_coord")};
}

Expr CompressedModeFormat::getPosArray(ModePack pack) const {
  return pack.getArray(0);
}

Expr CompressedModeFormat::getCoordArray(ModePack pack) const {
  return pack.getArray(1);
}

Expr CompressedModeFormat::getPosCapacity(Mode mode) const {
  const std::string varName = mode.getName() + "_pos_size";
 
  if (!mode.hasVar(varName)) {
    Expr posCapacity = Var::make(varName, Int());
    mode.addVar(varName, posCapacity);
    return posCapacity;
  }

  return mode.getVar(varName);
}

Expr CompressedModeFormat::getCoordCapacity(Mode mode) const {
  const std::string varName = mode.getName() + "_coord_size";
  
  if (!mode.hasVar(varName)) {
    Expr idxCapacity = Var::make(varName, Int());
    mode.addVar(varName, idxCapacity);
    return idxCapacity;
  }

  return mode.getVar(varName);
}

}
