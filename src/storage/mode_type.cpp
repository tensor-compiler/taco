#include <string>
#include <memory>
#include <vector>

#include "taco/storage/mode_type.h"
#include "taco/util/collections.h"
#include "taco/util/strings.h"

using namespace taco::ir;

namespace taco {

ModeType::ModeType() {
}

ModeType::ModeType(const std::shared_ptr<ModeTypeImpl> impl) : impl(impl) {
}

ModeType ModeType::operator()(const std::vector<Property>& properties) {
  return defined() ? impl->copy(properties) : ModeType();
}

bool ModeType::defined() const {
  return impl != nullptr;
}

std::string ModeType::getFormatName() const {
  return defined() ? impl->formatName : "undefined";
}

bool ModeType::isFull() const {
  taco_iassert(defined());
  return impl->isFull;
}

bool ModeType::isOrdered() const {
  taco_iassert(defined());
  return impl->isOrdered;
}

bool ModeType::isUnique() const {
  taco_iassert(defined());
  return impl->isUnique;
}

bool ModeType::isBranchless() const {
  taco_iassert(defined());
  return impl->isBranchless;
}

bool ModeType::isCompact() const {
  taco_iassert(defined());
  return impl->isCompact;
}

bool ModeType::hasCoordValIter() const {
  taco_iassert(defined());
  return impl->hasCoordValIter;
}

bool ModeType::hasCoordPosIter() const {
  taco_iassert(defined());
  return impl->hasCoordPosIter;
}

bool ModeType::hasLocate() const {
  taco_iassert(defined());
  return impl->hasLocate;
}

bool ModeType::hasInsert() const {
  taco_iassert(defined());
  return impl->hasInsert;
}

bool ModeType::hasAppend() const {
  taco_iassert(defined());
  return impl->hasAppend;
}

Mode::Mode(const ir::Expr tensor, const size_t mode, const Dimension size, 
           const ModePack* const pack, const size_t pos, 
           const ModeType prevModeType) :
    tensor(tensor), mode(mode), size(size), pack(pack), pos(pos), 
    prevModeType(prevModeType) {
}

std::string Mode::getName() const {
  return util::toString(tensor) + std::to_string(mode + 1);
}

ir::Expr Mode::getVar(const std::string varName) const {
  taco_iassert(hasVar(varName));
  return vars.at(varName);
}

bool Mode::hasVar(const std::string varName) const {
  return util::contains(vars, varName);
}

void Mode::addVar(const std::string varName, Expr var) {
  taco_iassert(isa<Var>(var));
  vars[varName] = var;
}

size_t ModePack::getSize() const {
  taco_iassert(modes.size() == modeTypes.size());
  return modes.size();
}

Expr ModePack::getArray(size_t idx) const {
  for (size_t i = 0; i < getSize(); ++i) {
    const auto arr = modeTypes[i].impl->getArray(idx, modes[i]);
    if (arr.defined()) {
      return arr;
    }
  }
  return Expr();
}

ModeTypeImpl::ModeTypeImpl(const std::string formatName, const bool isFull, 
                           const bool isOrdered, const bool isUnique, 
                           const bool isBranchless, const bool isCompact, 
                           const bool hasCoordValIter, 
                           const bool hasCoordPosIter, const bool hasLocate, 
                           const bool hasInsert, const bool hasAppend) : 
    formatName(formatName), isFull(isFull), isOrdered(isOrdered), 
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

bool operator==(const ModeType& a, const ModeType& b) {
  return (a.defined() && b.defined() && 
          a.getFormatName() == b.getFormatName() && 
          a.isFull() == b.isFull() && 
          a.isOrdered() == b.isOrdered() && 
          a.isUnique() == b.isUnique() && 
          a.isBranchless() == b.isBranchless() && 
          a.isCompact() == b.isCompact());
}

bool operator!=(const ModeType& a, const ModeType& b) {
  return !(a == b);
}

std::ostream& operator<<(std::ostream& os, const ModeType& modeType) {
  return os << modeType.getFormatName();
}

}

