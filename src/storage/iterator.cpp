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
struct Iterator::Content {
  old::TensorPath path;

  Mode     mode;
  Iterator parent;

  ir::Expr tensor;
  ir::Expr posVar;
  ir::Expr idxVar;
  ir::Expr endVar;
  ir::Expr segendVar;
  ir::Expr validVar;
  ir::Expr beginVar;
};

Iterator::Iterator() : content(nullptr) {
}

Iterator::Iterator(const ir::Expr& tensorVar) : content(new Content) {
  content->tensor = tensorVar;
  content->posVar = 0;
  content->idxVar = 0;
  content->endVar = 1;
}

Iterator::Iterator(const old::TensorPath& path, std::string indexVarName,
                   const ir::Expr& tensor, Mode mode, Iterator parent)
    : content(new Content) {
  content->path = path;

  content->mode = mode;
  content->parent = parent;

  string modeName = mode.getName();
  content->tensor = tensor;
  content->posVar = Var::make("p" + modeName, Int());
  content->idxVar = Var::make(indexVarName + util::toString(tensor), Int());
  content->endVar = Var::make(modeName + "_end", Int());
  content->segendVar = Var::make(modeName + "_segend", Int());
  content->validVar = Var::make("v" + modeName, Bool);
  content->beginVar = Var::make(modeName + "_begin", Int());
}

Iterator Iterator::make(std::string indexVarName, const ir::Expr& tensorVar,
                        Iterator parent, string name) {
  return Iterator();
}

const Iterator& Iterator::getParent() const {
  taco_iassert(defined());
  return content->parent;
}

const old::TensorPath& Iterator::getTensorPath() const {
  return content->path;
}

Expr Iterator::getTensor() const {
  taco_iassert(defined());
  return content->tensor;
}

const Mode& Iterator::getMode() const {
  taco_iassert(defined());
  return content->mode;
}

Expr Iterator::getPosVar() const {
  taco_iassert(defined());
  return content->posVar;
}

Expr Iterator::getIdxVar() const {
  taco_iassert(defined());
  return content->idxVar;
}

Expr Iterator::getIteratorVar() const {
  return hasCoordPosIter() ? getPosVar() : getIdxVar();
}

Expr Iterator::getDerivedVar() const {
  return hasCoordPosIter() ? getIdxVar() : getPosVar();
}

Expr Iterator::getEndVar() const {
  taco_iassert(defined());
  return content->endVar;
}

Expr Iterator::getSegendVar() const {
  taco_iassert(defined());
  return content->segendVar;
}

Expr Iterator::getValidVar() const {
  taco_iassert(defined());
  return content->validVar;
}

Expr Iterator::getBeginVar() const {
  taco_iassert(defined());
  return content->beginVar;
}

bool Iterator::isFull() const {
  taco_iassert(defined());
  return getMode().defined() && getMode().getModeType().isFull();
}

bool Iterator::isOrdered() const {
  taco_iassert(defined());
  return getMode().defined() && getMode().getModeType().isOrdered();
}

bool Iterator::isUnique() const {
  taco_iassert(defined());
  return getMode().defined() && getMode().getModeType().isUnique();
}

bool Iterator::isBranchless() const {
  taco_iassert(defined());
  return getMode().defined() && getMode().getModeType().isBranchless();
}

bool Iterator::isCompact() const {
  taco_iassert(defined());
  return getMode().defined() && getMode().getModeType().isCompact();
}

bool Iterator::hasCoordValIter() const {
  taco_iassert(defined());
  return getMode().defined() && getMode().getModeType().hasCoordValIter();
}

bool Iterator::hasCoordPosIter() const {
  taco_iassert(defined());
  return getMode().defined() && getMode().getModeType().hasCoordPosIter();
}

bool Iterator::hasLocate() const {
  taco_iassert(defined());
  return getMode().defined() && getMode().getModeType().hasLocate();
}

bool Iterator::hasInsert() const {
  taco_iassert(defined());
  return getMode().defined() && getMode().getModeType().hasInsert();
}

bool Iterator::hasAppend() const {
  taco_iassert(defined());
  return getMode().defined() && getMode().getModeType().hasAppend();
}

std::tuple<Stmt,Expr,Expr>
Iterator::getCoordIter(const std::vector<Expr>& coords) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeType().impl->getCoordIter(coords, getMode());
}

std::tuple<Stmt,Expr,Expr> Iterator::getCoordAccess(const Expr& pPrev, 
    const std::vector<Expr>& coords) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeType().impl->getCoordAccess(pPrev, coords, getMode());
}

std::tuple<Stmt,Expr,Expr> Iterator::getPosIter(const Expr& pPrev) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeType().impl->getPosIter(pPrev, getMode());
}

std::tuple<Stmt,Expr,Expr>
Iterator::getPosAccess(const Expr& p, const std::vector<Expr>& i) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeType().impl->getPosAccess(p, i, getMode());
}

std::tuple<Stmt,Expr,Expr>
Iterator::getLocate(const Expr& pPrev, const std::vector<Expr>& coord) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeType().impl->getLocate(pPrev, coord, getMode());
}

Stmt Iterator::getInsertCoord(const Expr& p, const std::vector<Expr>& coords) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeType().impl->getInsertCoord(p, coords, getMode());
}

Expr Iterator::getSize() const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeType().impl->getSize(getMode());
}

Stmt Iterator::getInsertInitCoords(const Expr& pBegin, const Expr& pEnd) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeType().impl->getInsertInitCoords(pBegin, pEnd,
                                                           getMode());
}

Stmt Iterator::getInsertInitLevel(const Expr& szPrev, const Expr& sz) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeType().impl->getInsertInitLevel(szPrev, sz,
                                                          getMode());
}

Stmt Iterator::getInsertFinalizeLevel(const Expr& szPrev, const Expr& sz) const{
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeType().impl->getInsertFinalizeLevel(szPrev, sz,
                                                              getMode());
}

Stmt Iterator::getAppendCoord(const Expr& p, const Expr& i) const {
  taco_iassert(defined() && content->mode.defined());
  return content->mode.getModeType().impl->getAppendCoord(p, i, content->mode);
}

Stmt Iterator::getAppendEdges(const Expr& pPrev, const Expr& pBegin, 
                              const Expr& pEnd) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeType().impl->getAppendEdges(pPrev, pBegin, pEnd,
                                                      getMode());
}

Stmt Iterator::getAppendInitEdges(const Expr& pPrevBegin, 
                                  const Expr& pPrevEnd) const {
  taco_iassert(defined() && content->mode.defined());
  return content->mode.getModeType().impl->getAppendInitEdges(pPrevBegin,
                                                              pPrevEnd,
                                                              content->mode);
}

Stmt Iterator::getAppendInitLevel(const Expr& szPrev, const Expr& sz) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeType().impl->getAppendInitLevel(szPrev, sz,
                                                          getMode());
}

Stmt Iterator::getAppendFinalizeLevel(const Expr& szPrev, const Expr& sz) const{
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeType().impl->getAppendFinalizeLevel(szPrev, sz,
                                                              getMode());
}

bool Iterator::defined() const {
  return content != nullptr;
}

bool operator==(const Iterator& a, const Iterator& b) {
  return a.content == b.content;
}

bool operator<(const Iterator& a, const Iterator& b) {
  return a.content < b.content;
}

std::ostream& operator<<(std::ostream& os, const Iterator& iterator) {
  if (!iterator.defined()) {
    return os << "Iterator()";
  }
  return os << util::toString(iterator.getTensor());
}

}
