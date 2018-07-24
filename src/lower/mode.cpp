#include "taco/lower/mode.h"

#include <string>
#include <memory>
#include <vector>

#include "taco/util/collections.h"
#include "taco/util/strings.h"

using namespace std;
using namespace taco::ir;

namespace taco {

// class Mode
struct Mode::Content {
  Expr            tensor;          /// the tensor containing mode
  Dimension       size;            /// the size of the mode
  size_t          level;           /// the location of mode in a mode hierarchy
  ModeType        modeType;        /// the type of the mode

  ModePack       modePack;        /// the pack that contains the mode
  size_t          packLoc;         /// position within pack containing mode

  ModeType        parentModeType;  /// type of previous mode in the tensor

  std::map<std::string, Expr> vars;
};

Mode::Mode() : content(nullptr) {
}

Mode::Mode(ir::Expr tensor, Dimension size, size_t level, ModeType modeType,
     ModePack modePack, size_t packLoc, ModeType parentModeType)
    : content(new Content) {
  taco_iassert(modeType.defined());
  content->tensor = tensor;
  content->size = size;
  content->level = level;
  content->modeType = modeType;
  content->modePack = modePack;
  content->packLoc = packLoc;
  content->parentModeType = parentModeType;
}

std::string Mode::getName() const {
  return util::toString(getTensorExpr()) + std::to_string(getLevel());
}

Expr Mode::getTensorExpr() const {
  return content->tensor;
}

Dimension Mode::getSize() const {
  return content->size;
}

size_t Mode::getLevel() const {
  return content->level;
}

ModeType Mode::getModeType() const {
  return content->modeType;
}

ModePack Mode::getModePack() const {
  return content->modePack;
}

size_t Mode::getPackLocation() const {
  return content->packLoc;
}

ModeType Mode::getParentModeType() const {
  return content->parentModeType;
}

Expr Mode::getVar(std::string varName) const {
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

bool Mode::defined() const {
  return content != nullptr;
}

std::ostream& operator<<(std::ostream& os, const Mode& mode) {
  return os << mode.getName();
}


// class ModePack
struct ModePack::Content {
  size_t numModes = 0;
  vector<Expr> arrays;
};

ModePack::ModePack() : content(new Content) {
}

ModePack::ModePack(size_t numModes, ModeType modeType, ir::Expr tensor,
                     size_t level) : ModePack() {
  content->numModes = numModes;
  content->arrays = modeType.impl->getArrays(tensor, level);
}

size_t ModePack::getNumModes() const {
  return content->numModes;
}

ir::Expr ModePack::getArray(size_t i) const {
  return content->arrays[i];
}


// class ModeFunction
struct ModeFunction::Content {
  Stmt body;
  vector<Expr> results;
};

ModeFunction::ModeFunction(Stmt body, const vector<Expr>& results)
    : content(new Content) {
  content->body = body;
  content->results = results;
}

ModeFunction::ModeFunction() : content(nullptr) {
}

ir::Stmt ModeFunction::getBody() const {
  return content->body;
}

bool ModeFunction::hasBody() const {
  return content->body.defined();
}

const std::vector<ir::Expr>& ModeFunction::getResults() const {
  return content->results;
}

bool ModeFunction::defined() const {
  return content != nullptr;
}

std::ostream& operator<<(std::ostream& os, const ModeFunction& modeFunction) {
  return os << modeFunction.getBody() << endl
            << util::join(modeFunction.getResults());
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

ModeFunction ModeTypeImpl::coordBounds(vector<Expr> coords, Mode mode) const {
  return ModeFunction();
}

ModeFunction ModeTypeImpl::coordAccess(ir::Expr parentPos,
                                       std::vector<ir::Expr> coords,
                                       Mode mode) const {
  return ModeFunction();
}

ModeFunction ModeTypeImpl::posBounds(ir::Expr parentPos, Mode mode) const {
  return ModeFunction();
}

ModeFunction ModeTypeImpl::posAccess(ir::Expr pos, std::vector<ir::Expr> coords,
                                     Mode mode) const {
  return ModeFunction();
}

ModeFunction ModeTypeImpl::locate(ir::Expr parentPos,
                                  std::vector<ir::Expr> coords,
                                  Mode mode) const {
  return ModeFunction();
}
  
Stmt ModeTypeImpl::getInsertCoord(Expr p,
    const std::vector<Expr>& i, Mode mode) const {
  return Stmt();
}

Expr ModeTypeImpl::getSize(Mode mode) const {
  return Expr();
}

Stmt ModeTypeImpl::getInsertInitCoords(Expr pBegin,
    Expr pEnd, Mode mode) const {
  return Stmt();
}

Stmt ModeTypeImpl::getInsertInitLevel(Expr szPrev,
    Expr sz, Mode mode) const {
  return Stmt();
}

Stmt ModeTypeImpl::getInsertFinalizeLevel(Expr szPrev,
    Expr sz, Mode mode) const {
  return Stmt();
}

Stmt ModeTypeImpl::getAppendCoord(Expr p, Expr i,
    Mode mode) const {
  return Stmt();
}

Stmt ModeTypeImpl::getAppendEdges(Expr pPrev, Expr pBegin,
    Expr pEnd, Mode mode) const {
  return Stmt();
}

Stmt ModeTypeImpl::getAppendInitEdges(Expr pPrevBegin,
    Expr pPrevEnd, Mode mode) const {
  return Stmt();
}

Stmt ModeTypeImpl::getAppendInitLevel(Expr szPrev,
    Expr sz, Mode mode) const {
  return Stmt();
}

Stmt ModeTypeImpl::getAppendFinalizeLevel(Expr szPrev,
    Expr sz, Mode mode) const {
  return Stmt();
}

}
