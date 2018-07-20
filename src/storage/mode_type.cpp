#include <string>
#include <memory>
#include <vector>

#include "taco/storage/mode_type.h"
#include "taco/util/collections.h"
#include "taco/util/strings.h"

using namespace std;
using namespace taco::ir;

namespace taco {

// class Mode
struct Mode::Content {
  ir::Expr        tensor;        /// the tensor containing mode
  size_t          level;         /// the location of mode in a mode hierarchy
  Dimension       size;          /// the size of the mode

  const ModePack* pack;          /// the pack that contains the mode
  size_t          packLoc;       /// position within pack containing mode

  ModeType        prevModeType;  /// type of previous mode in containing tensor

  std::map<std::string, ir::Expr> vars;
};

Mode::Mode(ir::Expr tensor, size_t level, Dimension size, const ModePack* pack,
           size_t packLoc, ModeType prevModeType) : content(new Content) {
  content->tensor = tensor;
  content->level = level;
  content->size = size;
  content->pack = pack;
  content->packLoc = packLoc;
  content->prevModeType = prevModeType;
}

std::string Mode::getName() const {
  return util::toString(getTensorExpr()) + std::to_string(getLevel()+1);
}

ir::Expr Mode::getTensorExpr() const {
  return content->tensor;
}

size_t Mode::getLevel() const {
  return content->level;
}

Dimension Mode::getSize() const {
  return content->size;
}

const ModePack* Mode::getPack() const {
  return content->pack;
}

size_t Mode::getPackLocation() const {
  return content->packLoc;
}

ModeType Mode::getParentModeType() const {
  return content->prevModeType;
}

ir::Expr Mode::getVar(std::string varName) const {
  taco_iassert(hasVar(varName));
  return content->vars.at(varName);
}

bool Mode::hasVar(std::string varName) const {
  return util::contains(content->vars, varName);
}

void Mode::addVar(std::string varName, Expr var) {
  taco_iassert(isa<Var>(var));
  content->vars[varName] = var;
}


// class ModePack
size_t ModePack::getSize() const {
  taco_iassert(modes.size() == modeTypes.size());
  return modes.size();
}

Expr ModePack::getArray(size_t i) const {
  for (size_t j = 0; j < getSize(); ++j) {
    const auto arr = modeTypes[j].impl->getArray(i, modes[j]);
    if (arr.defined()) {
      return arr;
    }
  }
  return Expr();
}

// class ModeTypeImpl
ModeTypeImpl::ModeTypeImpl(const std::string name,
                           bool isFull, bool isOrdered, bool isUnique,
                           bool isBranchless, bool isCompact,
                           bool hasCoordValIter, bool hasCoordPosIter,
                           bool hasLocate, bool hasInsert, bool hasAppend) :
    name(name), isFull(isFull), isOrdered(isOrdered),
    isUnique(isUnique), isBranchless(isBranchless), isCompact(isCompact), 
    hasCoordValIter(hasCoordValIter), hasCoordPosIter(hasCoordPosIter), 
    hasLocate(hasLocate), hasInsert(hasInsert), hasAppend(hasAppend) {
}

std::tuple<Stmt,Expr,Expr> ModeTypeImpl::getCoordIter(
    const std::vector<Expr>& i, Mode& mode) const {
  return std::tuple<Stmt,Expr,Expr>(Stmt(), Expr(), Expr());
}

std::tuple<Stmt,Expr,Expr> ModeTypeImpl::getCoordAccess(const Expr& pPrev, 
    const std::vector<Expr>& i, Mode& mode) const {
  return std::tuple<Stmt,Expr,Expr>(Stmt(), Expr(), Expr());
}

std::tuple<Stmt,Expr,Expr> ModeTypeImpl::getPosIter(const Expr& pPrev, 
    Mode& mode) const {
  return std::tuple<Stmt,Expr,Expr>(Stmt(), Expr(), Expr());
}

std::tuple<Stmt,Expr,Expr> ModeTypeImpl::getPosAccess(const Expr& p, 
    const std::vector<Expr>& i, Mode& mode) const {
  return std::tuple<Stmt,Expr,Expr>(Stmt(), Expr(), Expr());
}

std::tuple<Stmt,Expr,Expr> ModeTypeImpl::getLocate(const Expr& pPrev, 
    const std::vector<Expr>& i, Mode& mode) const {
  return std::tuple<Stmt,Expr,Expr>(Stmt(), Expr(), Expr());
}
  
Stmt ModeTypeImpl::getInsertCoord(const ir::Expr& p, 
    const std::vector<ir::Expr>& i, Mode& mode) const {
  return Stmt();
}

Expr ModeTypeImpl::getSize(Mode& mode) const {
  return Expr();
}

Stmt ModeTypeImpl::getInsertInitCoords(const ir::Expr& pBegin, 
    const ir::Expr& pEnd, Mode& mode) const {
  return Stmt();
}

Stmt ModeTypeImpl::getInsertInitLevel(const ir::Expr& szPrev, 
    const ir::Expr& sz, Mode& mode) const {
  return Stmt();
}

Stmt ModeTypeImpl::getInsertFinalizeLevel(const ir::Expr& szPrev, 
    const ir::Expr& sz, Mode& mode) const {
  return Stmt();
}

Stmt ModeTypeImpl::getAppendCoord(const ir::Expr& p, const ir::Expr& i, 
    Mode& mode) const {
  return Stmt();
}

Stmt ModeTypeImpl::getAppendEdges(const ir::Expr& pPrev, const ir::Expr& pBegin, 
    const ir::Expr& pEnd, Mode& mode) const {
  return Stmt();
}

Stmt ModeTypeImpl::getAppendInitEdges(const ir::Expr& pPrevBegin, 
    const ir::Expr& pPrevEnd, Mode& mode) const {
  return Stmt();
}

Stmt ModeTypeImpl::getAppendInitLevel(const ir::Expr& szPrev, 
    const ir::Expr& sz, Mode& mode) const {
  return Stmt();
}

Stmt ModeTypeImpl::getAppendFinalizeLevel(const ir::Expr& szPrev, 
    const ir::Expr& sz, Mode& mode) const {
  return Stmt();
}

}

