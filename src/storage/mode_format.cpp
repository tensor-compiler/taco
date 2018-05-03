#include <string>
#include <memory>
#include <vector>

#include "taco/storage/mode_format.h"
#include "taco/util/strings.h"

using namespace taco::ir;

namespace taco {

std::string ModeType::Mode::getName() const {
  return util::toString(tensor) + std::to_string(mode + 1);
}

ModeType::ModePack::ModePack(const std::vector<Mode>& modes, 
                             const std::vector<ModeType>& modeTypes) : 
    modes(modes), modeTypes(modeTypes) {
  taco_iassert(modes.size() == modeTypes.size());
}

Expr ModeType::ModePack::getArray(size_t idx) const {
  for (size_t i = 0; i < (size_t)getSize(); ++i) {
    const auto arr = modeTypes[i].impl->getArray(idx, modes[i]);
    if (arr.defined()) {
      return arr;
    }
  }
  return Expr();
}

ModeType::ModeType(const std::shared_ptr<ModeFormat> modeFormat) : 
    impl(modeFormat) {}

ModeType::ModeType(const ModeType& modeType) : impl(modeType.impl) {}

ModeType& ModeType::operator=(const ModeType& modeType) {
  impl = modeType.impl;
  return *this;
}
ModeType ModeType::operator()(const std::vector<Property>& properties) {
  return impl->copy(properties);
}

std::string ModeType::getFormatName() const {
  return impl->formatName;
}

bool ModeType::isFull() const {
  return impl->isFull;
}

bool ModeType::isOrdered() const {
  return impl->isOrdered;
}

bool ModeType::isUnique() const {
  return impl->isUnique;
}

bool ModeType::isBranchless() const {
  return impl->isBranchless;
}

bool ModeType::isCompact() const {
  return impl->isCompact;
}

bool ModeType::hasCoordValueIter() const {
  return impl->hasCoordValueIter;
}

bool ModeType::hasCoordPosIter() const {
  return impl->hasCoordPosIter;
}

bool ModeType::hasLocate() const {
  return impl->hasLocate;
}

bool ModeType::hasInsert() const {
  return impl->hasInsert;
}

bool ModeType::hasAppend() const {
  return impl->hasAppend;
}

ModeFormat::ModeFormat(const std::string formatName, const bool isFull, 
                       const bool isOrdered, const bool isUnique, 
                       const bool isBranchless, const bool isCompact, 
                       const bool hasCoordValueIter, const bool hasCoordPosIter, 
                       const bool hasLocate, const bool hasInsert, 
                       const bool hasAppend) : 
    formatName(formatName), isFull(isFull), isOrdered(isOrdered), 
    isUnique(isUnique), isBranchless(isBranchless), isCompact(isCompact), 
    hasCoordValueIter(hasCoordValueIter), hasCoordPosIter(hasCoordPosIter), 
    hasLocate(hasLocate), hasInsert(hasInsert), hasAppend(hasAppend) {}

std::tuple<Stmt,Expr,Expr> ModeFormat::getCoordIter(const std::vector<Expr>& i, 
    const ModeType::Mode& mode) const {
  return std::tuple<Stmt,Expr,Expr>(Stmt(), Expr(), Expr());
}

std::tuple<Stmt,Expr,Expr> ModeFormat::getCoordAccess(const Expr& pPrev, 
    const std::vector<Expr>& i, const ModeType::Mode& mode) const {
  return std::tuple<Stmt,Expr,Expr>(Stmt(), Expr(), Expr());
}

std::tuple<Stmt,Expr,Expr> ModeFormat::getPosIter(const Expr& pPrev, 
    const ModeType::Mode& mode) const {
  return std::tuple<Stmt,Expr,Expr>(Stmt(), Expr(), Expr());
}

std::tuple<Stmt,Expr,Expr> ModeFormat::getPosAccess(const Expr& p, 
    const std::vector<Expr>& i, const ModeType::Mode& mode) const {
  return std::tuple<Stmt,Expr,Expr>(Stmt(), Expr(), Expr());
}

std::tuple<Stmt,Expr,Expr> ModeFormat::getLocate(const Expr& pPrev, 
    const std::vector<Expr>& i, const ModeType::Mode& mode) const {
  return std::tuple<Stmt,Expr,Expr>(Stmt(), Expr(), Expr());
}

Stmt ModeFormat::getInsertCoord(const Expr& p, const std::vector<Expr>& i, 
    const ModeType::Mode& mode) const {
  return Stmt();
}

Expr ModeFormat::getSize(const Expr& szPrev, const ModeType::Mode& mode) const {
  return Expr();
}

Stmt ModeFormat::getInsertInit(const Expr& szPrev, const Expr& sz, 
    const ModeType::Mode& mode) const {
  return Stmt();
}

Stmt ModeFormat::getInsertFinalize(const Expr& szPrev, const Expr& sz, 
    const ModeType::Mode& mode) const {
  return Stmt();
}

Stmt ModeFormat::getAppendCoord(const Expr& p, const std::vector<Expr>& i, 
    const ModeType::Mode& mode) const {
  return Stmt();
}

Stmt ModeFormat::getAppendEdges(const Expr& pPrev, const Expr& pBegin, 
    const Expr& pEnd, const ModeType::Mode& mode) const {
  return Stmt();
}

Stmt ModeFormat::getAppendInit(const Expr& szPrev, const Expr& sz, 
    const ModeType::Mode& mode) const {
  return Stmt();
}

Stmt ModeFormat::getAppendFinalize(const Expr& szPrev, const Expr& sz, 
    const ModeType::Mode& mode) const {
  return Stmt();
}

bool operator==(const ModeType& a, const ModeType& b) {
  return (a.getFormatName() == b.getFormatName());
}

bool operator!=(const ModeType& a, const ModeType& b) {
  return !(a == b);
}

std::ostream& operator<<(std::ostream& os, const ModeType& modeType) {
  return os << modeType.getFormatName();
}

}

