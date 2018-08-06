#include "iterator.h"

#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "mode_access.h"
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
  ir::Expr coordVar;
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
  content->coordVar = 0;
  content->endVar = 1;
}

Iterator::Iterator(const old::TensorPath& path, std::string coordVarName,
                   const ir::Expr& tensor, Mode mode, Iterator parent)
    : content(new Content) {
  content->path = path;

  content->mode = mode;
  content->parent = parent;

  string modeName = mode.getName();
  content->tensor = tensor;
  content->posVar = Var::make("p" + modeName, Int());
  content->coordVar = Var::make(coordVarName + util::toString(tensor), Int());
  content->endVar = Var::make(modeName + "_end", Int());
  content->segendVar = Var::make(modeName + "_segend", Int());
  content->validVar = Var::make("v" + modeName, Bool);
  content->beginVar = Var::make(modeName + "_begin", Int());
}

Iterator::Iterator(Expr tensor, Mode mode, Iterator parent, string name)
    : content(new Content) {
  content->mode = mode;
  content->parent = parent;

  string modeName = mode.getName();
  content->tensor = tensor;
  content->posVar = Var::make("p" + modeName, Int());
  content->coordVar = Var::make(name, Int());
  content->endVar = Var::make(modeName + "_end", Int());
  content->segendVar = Var::make(modeName + "_segend", Int());
  content->validVar = Var::make("v" + modeName, Bool);
  content->beginVar = Var::make(modeName + "_begin", Int());
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

Expr Iterator::getCoordVar() const {
  taco_iassert(defined());
  return content->coordVar;
}

Expr Iterator::getIteratorVar() const {
  return hasPosIter() ? getPosVar() : getCoordVar();
}

Expr Iterator::getDerivedVar() const {
  return hasPosIter() ? getCoordVar() : getPosVar();
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

bool Iterator::hasCoordIter() const {
  taco_iassert(defined());
  return getMode().defined() && getMode().getModeType().hasCoordValIter();
}

bool Iterator::hasPosIter() const {
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

ModeFunction Iterator::coordBounds(const std::vector<ir::Expr>& coords) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeType().impl->coordIterBounds(coords, getMode());
}

ModeFunction Iterator::coordAccess(const std::vector<ir::Expr>& coords) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeType().impl->coordIterAccess(getParent().getPosVar(),
                                                   coords, getMode());
}

ModeFunction Iterator::posBounds() const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeType().impl->posIterBounds(getParent().getPosVar(),
                                               getMode());
}

ModeFunction Iterator::posAccess(const std::vector<ir::Expr>& coords) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeType().impl->posIterAccess(getPosVar(),
                                                 coords, getMode());
}

ModeFunction Iterator::locate(const std::vector<ir::Expr>& coords) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeType().impl->locate(getParent().getPosVar(),
                                              coords, getMode());
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

void createIterators(IndexStmt stmt,
                     const std::map<TensorVar, ir::Expr>& tensorVars,
                     std::map<ModeAccess, Iterator>* iterators,
                     std::map<Iterator, IndexVar>* indexVars,
                     std::map<IndexVar, ir::Expr>* coordVars) {
  taco_iassert(iterators != nullptr);
  taco_iassert(indexVars != nullptr);
  taco_iassert(coordVars != nullptr);
  match(stmt,
    function<void(const AccessNode*)>([&](const AccessNode* n) {
      taco_iassert(util::contains(tensorVars, n->tensorVar));
      Expr tensorVarIR = tensorVars.at(n->tensorVar);
      Shape shape = n->tensorVar.getType().getShape();
      Format format = n->tensorVar.getFormat();
      set<IndexVar> vars(n->indexVars.begin(), n->indexVars.end());

      Iterator parent(tensorVarIR);
      iterators->insert({{Access(n),0}, parent});

      size_t level = 1;
      ModeFormat parentModeType;
      for (ModeFormatPack modeTypePack : format.getModeTypePacks()) {
        vector<Expr> arrays;
        taco_iassert(modeTypePack.getModeTypes().size() > 0);

        ModePack modePack(modeTypePack.getModeTypes().size(),
                          modeTypePack.getModeTypes()[0], tensorVarIR, level);

        int pos = 0;
        for (auto& modeType : modeTypePack.getModeTypes()) {
          size_t modeNumber = format.getModeOrdering()[level-1];
          Dimension dim = shape.getDimension(modeNumber);
          IndexVar indexVar = n->indexVars[modeNumber];
          Mode mode(tensorVarIR, dim, level, modeType, modePack, pos,
                    parentModeType);

          string name = indexVar.getName() + n->tensorVar.getName();
          Iterator iterator(tensorVarIR, mode, parent, name);
          iterators->insert({{Access(n),level}, iterator});
          indexVars->insert({iterator, indexVar});

          parent = iterator;
          parentModeType = modeType;
          pos++;
          level++;
        }
      }
    }),
    function<void(const ForallNode*, Matcher*)>([&](const ForallNode* n,
                                                    Matcher* m) {
      Expr coord = Var::make(n->indexVar.getName(), type<int32_t>());
      coordVars->insert({n->indexVar, coord});
      m->match(n->stmt);
    }),
    function<void(const AssignmentNode*,Matcher*)>([&](const AssignmentNode* n,
                                                       Matcher* m) {
      m->match(n->rhs);
      m->match(n->lhs);
    })
  );
}

std::vector<Iterator> getAppenders(const std::vector<Iterator>& iterators) {
  vector<Iterator> appendIterators;
  for (auto& iterator : iterators) {
    if (iterator.hasAppend()) {
      appendIterators.push_back(iterator);
    }
  }
  return appendIterators;
}

std::vector<Iterator> getInserters(const std::vector<Iterator>& iterators) {
 vector<Iterator> result;
  for (auto& iterator : iterators) {
    if (iterator.hasInsert()) {
      taco_iassert(iterator.hasLocate())
          << "Iterators with insert must also have locate";
      result.push_back(iterator);
    }
  }
  return result;
}

}
