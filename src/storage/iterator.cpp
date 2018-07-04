#include "iterator.h"

#include "taco/index_notation/index_notation.h"
#include "taco/ir/ir.h"
#include "taco/storage/storage.h"
#include "taco/storage/array.h"
#include "taco/util/strings.h"

using namespace std;
using namespace taco::ir;

namespace taco {

// class Iterator
Iterator::Iterator() : iterator(nullptr) {
}

Iterator Iterator::makeRoot(const Expr& tensorVar) {
  Iterator iterator;
  iterator.iterator = std::make_shared<IteratorImpl>(tensorVar);
  return iterator;
}

Iterator Iterator::make(const old::TensorPath& path, std::string indexVarName,
                        const Expr& tensorVar, ModeType modeType, Mode* mode, 
                        Iterator parent) {
  Iterator iterator;
  iterator.path = path;
  iterator.iterator = std::make_shared<IteratorImpl>(parent, indexVarName, 
                                                     tensorVar, modeType, mode);
  taco_iassert(iterator.defined());
  return iterator;
}

Iterator Iterator::make(std::string indexVarName, const ir::Expr& tensorVar,
                        ModeType modeType, Iterator parent, string name) {
  Iterator iterator;
  iterator.iterator = std::make_shared<IteratorImpl>(parent, indexVarName,
                                                     tensorVar, modeType, name);
  taco_iassert(iterator.defined());
  return iterator;
}

const Iterator& Iterator::getParent() const {
  taco_iassert(defined());
  return iterator->getParent();
}

const old::TensorPath& Iterator::getTensorPath() const {
  return path;
}

Expr Iterator::getTensor() const {
  taco_iassert(defined());
  return iterator->getTensor();
}

const Mode& Iterator::getMode() const {
  taco_iassert(defined());
  return iterator->getMode();
}

Expr Iterator::getPosVar() const {
  taco_iassert(defined());
  return iterator->getPosVar();
}

Expr Iterator::getIdxVar() const {
  taco_iassert(defined());
  return iterator->getIdxVar();
}

Expr Iterator::getIteratorVar() const {
  return hasCoordPosIter() ? getPosVar() : getIdxVar();
}

Expr Iterator::getDerivedVar() const {
  return hasCoordPosIter() ? getIdxVar() : getPosVar();
}

Expr Iterator::getEndVar() const {
  taco_iassert(defined());
  return iterator->getEndVar();
}

Expr Iterator::getSegendVar() const {
  taco_iassert(defined());
  return iterator->getSegendVar();
}

Expr Iterator::getValidVar() const {
  taco_iassert(defined());
  return iterator->getValidVar();
}

Expr Iterator::getBeginVar() const {
  taco_iassert(defined());
  return iterator->getBeginVar();
}

bool Iterator::isFull() const {
  taco_iassert(defined());
  return iterator->isFull();
}

bool Iterator::isOrdered() const {
  taco_iassert(defined());
  return iterator->isOrdered();
}

bool Iterator::isUnique() const {
  taco_iassert(defined());
  return iterator->isUnique();
}

bool Iterator::isBranchless() const {
  taco_iassert(defined());
  return iterator->isBranchless();
}

bool Iterator::isCompact() const {
  taco_iassert(defined());
  return iterator->isCompact();
}

bool Iterator::hasCoordValIter() const {
  taco_iassert(defined());
  return iterator->hasCoordValIter();
}

bool Iterator::hasCoordPosIter() const {
  taco_iassert(defined());
  return iterator->hasCoordPosIter();
}

bool Iterator::hasLocate() const {
  taco_iassert(defined());
  return iterator->hasLocate();
}

bool Iterator::hasInsert() const {
  taco_iassert(defined());
  return iterator->hasInsert();
}

bool Iterator::hasAppend() const {
  taco_iassert(defined());
  return iterator->hasAppend();
}

std::tuple<Stmt,Expr,Expr> Iterator::getCoordIter(
    const std::vector<Expr>& i) const {
  taco_iassert(defined());
  return iterator->getCoordIter(i);
}

std::tuple<Stmt,Expr,Expr> Iterator::getCoordAccess(const Expr& pPrev, 
    const std::vector<Expr>& i) const {
  taco_iassert(defined());
  return iterator->getCoordAccess(pPrev, i);
}

std::tuple<Stmt,Expr,Expr> Iterator::getPosIter(const Expr& pPrev) const {
  taco_iassert(defined());
  return iterator->getPosIter(pPrev);
}

std::tuple<Stmt,Expr,Expr> Iterator::getPosAccess(const Expr& p, 
    const std::vector<Expr>& i) const {
  taco_iassert(defined());
  return iterator->getPosAccess(p, i);
}

std::tuple<Stmt,Expr,Expr> Iterator::getLocate(const Expr& pPrev, 
    const std::vector<Expr>& i) const {
  taco_iassert(defined());
  return iterator->getLocate(pPrev, i);
}

Stmt Iterator::getInsertCoord(const Expr& p, const std::vector<Expr>& i) const {
  taco_iassert(defined());
  return iterator->getInsertCoord(p, i);
}

Expr Iterator::getSize() const {
  taco_iassert(defined());
  return iterator->getSize();
}

Stmt Iterator::getInsertInitCoords(const Expr& pBegin, const Expr& pEnd) const {
  taco_iassert(defined());
  return iterator->getInsertInitCoords(pBegin, pEnd);
}

Stmt Iterator::getInsertInitLevel(const Expr& szPrev, const Expr& sz) const {
  taco_iassert(defined());
  return iterator->getInsertInitLevel(szPrev, sz);
}

Stmt Iterator::getInsertFinalizeLevel(const Expr& szPrev, 
    const Expr& sz) const {
  taco_iassert(defined());
  return iterator->getInsertFinalizeLevel(szPrev, sz);
}

Stmt Iterator::getAppendCoord(const Expr& p, const Expr& i) const {
  taco_iassert(defined());
  return iterator->getAppendCoord(p, i);
}

Stmt Iterator::getAppendEdges(const Expr& pPrev, const Expr& pBegin, 
    const Expr& pEnd) const {
  taco_iassert(defined());
  return iterator->getAppendEdges(pPrev, pBegin, pEnd);
}

Stmt Iterator::getAppendInitEdges(const Expr& pPrevBegin, 
    const Expr& pPrevEnd) const {
  taco_iassert(defined());
  return iterator->getAppendInitEdges(pPrevBegin, pPrevEnd);
}

Stmt Iterator::getAppendInitLevel(const Expr& szPrev, const Expr& sz) const {
  taco_iassert(defined());
  return iterator->getAppendInitLevel(szPrev, sz);
}

Stmt Iterator::getAppendFinalizeLevel(const Expr& szPrev, 
    const Expr& sz) const {
  taco_iassert(defined());
  return iterator->getAppendFinalizeLevel(szPrev, sz);
}

bool Iterator::defined() const {
  return iterator != nullptr;
}

bool operator==(const Iterator& a, const Iterator& b) {
  return a.iterator == b.iterator;
}

bool operator<(const Iterator& a, const Iterator& b) {
  return a.iterator < b.iterator;
}

std::ostream& operator<<(std::ostream& os, const Iterator& iterator) {
  if (!iterator.defined()) {
    return os << "Iterator()";
  }
  return os << *iterator.iterator;
}

// class IteratorImpl
IteratorImpl::IteratorImpl(const ir::Expr& tensorVar) : 
    tensorVar(tensorVar), posVar(0ll), idxVar(0ll), endVar(1ll) {
}

IteratorImpl::IteratorImpl(Iterator parent, std::string indexVarName, 
                           const ir::Expr& tensorVar, ModeType modeType, 
                           Mode* mode)
    : IteratorImpl(parent, indexVarName, tensorVar, modeType, mode->getName()){
  this->mode = mode;
}

IteratorImpl::IteratorImpl(Iterator parent, string indexVarName,
                           const ir::Expr& tensorVar, ModeType modeType,
                           string modeName) :
    parent(parent), tensorVar(tensorVar),
    posVar(Var::make("p" + modeName, Int())),
    idxVar(Var::make(indexVarName + util::toString(tensorVar), Int())),
    endVar(Var::make(modeName + "_end", Int())),
    segendVar(Var::make(modeName + "_segend", Int())),
    validVar(Var::make("v" + modeName, Bool)),
    beginVar(Var::make(modeName + "_begin", Int())),
    modeType(modeType) {
}

std::string IteratorImpl::getName() const {
  return util::toString(tensorVar);
}

const Iterator& IteratorImpl::getParent() const {
  return parent;
}

const Expr& IteratorImpl::getTensor() const {
  return tensorVar;
}

const Mode& IteratorImpl::getMode() const {
  taco_iassert(mode != nullptr);
  return *mode;
}

Expr IteratorImpl::getPosVar() const {
  return posVar;
}

Expr IteratorImpl::getIdxVar() const {
  return idxVar;
}

Expr IteratorImpl::getEndVar() const {
  return endVar;
}

Expr IteratorImpl::getSegendVar() const {
  return segendVar;
}

Expr IteratorImpl::getValidVar() const {
  return validVar;
}

Expr IteratorImpl::getBeginVar() const {
  return beginVar;
}

bool IteratorImpl::isFull() const {
  return modeType.defined() && modeType.isFull();
}

bool IteratorImpl::isOrdered() const {
  return modeType.defined() && modeType.isOrdered();
}

bool IteratorImpl::isUnique() const {
  return modeType.defined() && modeType.isUnique();
}

bool IteratorImpl::isBranchless() const {
  return modeType.defined() && modeType.isBranchless();
}

bool IteratorImpl::isCompact() const {
  return modeType.defined() && modeType.isCompact();
}

bool IteratorImpl::hasCoordValIter() const {
  return modeType.defined() && modeType.hasCoordValIter();
}

bool IteratorImpl::hasCoordPosIter() const {
  return modeType.defined() && modeType.hasCoordPosIter();
}

bool IteratorImpl::hasLocate() const {
  return modeType.defined() && modeType.hasLocate();
}

bool IteratorImpl::hasInsert() const {
  return modeType.defined() && modeType.hasInsert();
}

bool IteratorImpl::hasAppend() const {
  return modeType.defined() && modeType.hasAppend();
}

std::tuple<Stmt,Expr,Expr> IteratorImpl::getCoordIter(
    const std::vector<Expr>& i) const {
  taco_iassert(modeType.defined());
  return modeType.impl->getCoordIter(i, *mode);
}

std::tuple<Stmt,Expr,Expr> IteratorImpl::getCoordAccess(const Expr& pPrev, 
    const std::vector<Expr>& i) const {
  taco_iassert(modeType.defined());
  return modeType.impl->getCoordAccess(pPrev, i, *mode);
}

std::tuple<Stmt,Expr,Expr> IteratorImpl::getPosIter(const Expr& pPrev) const {
  taco_iassert(modeType.defined());
  return modeType.impl->getPosIter(pPrev, *mode);
}

std::tuple<Stmt,Expr,Expr> IteratorImpl::getPosAccess(const Expr& p, 
    const std::vector<Expr>& i) const {
  taco_iassert(modeType.defined());
  return modeType.impl->getPosAccess(p, i, *mode);
}

std::tuple<Stmt,Expr,Expr> IteratorImpl::getLocate(const Expr& pPrev, 
    const std::vector<Expr>& i) const {
  taco_iassert(modeType.defined());
  return modeType.impl->getLocate(pPrev, i, *mode);
}

Stmt IteratorImpl::getInsertCoord(const Expr& p, 
    const std::vector<Expr>& i) const {
  taco_iassert(modeType.defined());
  return modeType.impl->getInsertCoord(p, i, *mode);
}

Expr IteratorImpl::getSize() const {
  taco_iassert(modeType.defined());
  return modeType.impl->getSize(*mode);
}

Stmt IteratorImpl::getInsertInitCoords(const Expr& pBegin, 
    const Expr& pEnd) const {
  taco_iassert(modeType.defined());
  return modeType.impl->getInsertInitCoords(pBegin, pEnd, *mode);
}

Stmt IteratorImpl::getInsertInitLevel(const Expr& szPrev, 
    const Expr& sz) const {
  taco_iassert(modeType.defined());
  return modeType.impl->getInsertInitLevel(szPrev, sz, *mode);
}

Stmt IteratorImpl::getInsertFinalizeLevel(const Expr& szPrev, 
    const Expr& sz) const {
  taco_iassert(modeType.defined());
  return modeType.impl->getInsertFinalizeLevel(szPrev, sz, *mode);
}

Stmt IteratorImpl::getAppendCoord(const Expr& p, const Expr& i) const {
  taco_iassert(modeType.defined());
  return modeType.impl->getAppendCoord(p, i, *mode);
}

Stmt IteratorImpl::getAppendEdges(const Expr& pPrev, const Expr& pBegin, 
    const Expr& pEnd) const {
  taco_iassert(modeType.defined());
  return modeType.impl->getAppendEdges(pPrev, pBegin, pEnd, *mode);
}

Stmt IteratorImpl::getAppendInitEdges(const Expr& pPrevBegin, 
    const Expr& pPrevEnd) const {
  taco_iassert(modeType.defined());
  return modeType.impl->getAppendInitEdges(pPrevBegin, pPrevEnd, *mode);
}

Stmt IteratorImpl::getAppendInitLevel(const Expr& szPrev, 
    const Expr& sz) const {
  taco_iassert(modeType.defined());
  return modeType.impl->getAppendInitLevel(szPrev, sz, *mode);
}

Stmt IteratorImpl::getAppendFinalizeLevel(const Expr& szPrev, 
    const Expr& sz) const {
  taco_iassert(modeType.defined());
  return modeType.impl->getAppendFinalizeLevel(szPrev, sz, *mode);
}

std::ostream& operator<<(std::ostream& os, const IteratorImpl& iterator) {
  return os << iterator.getName();
}

}
