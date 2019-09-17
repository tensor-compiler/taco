#include "taco/lower/iterator.h"

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
  IndexVar indexVar;
  Mode     mode;

  Iterator               parent;  // Pointer to parent iterator
  std::weak_ptr<Content> child;   // (Non-reference counted) pointer to child iterator

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

Iterator::Iterator(std::shared_ptr<Content> content) : content(content) {
}

Iterator::Iterator(IndexVar indexVar) : content(new Content) {
  content->indexVar = indexVar;
  content->coordVar = Var::make(indexVar.getName(), Int());
}

Iterator::Iterator(ir::Expr tensor) : content(new Content) {
  content->tensor = tensor;
  content->posVar = 0;
  content->coordVar = 0;
  content->endVar = 1;
}

Iterator::Iterator(IndexVar indexVar, Expr tensor, Mode mode, Iterator parent,
                   string name) : content(new Content) {
  content->indexVar = indexVar;

  content->mode = mode;
  content->parent = parent;
  content->parent.setChild(*this);

  string modeName = mode.getName();
  content->tensor = tensor;

  content->posVar   = Var::make("p" + modeName,            Int());
  content->endVar   = Var::make("p" + modeName + "_end",   Int());
  content->beginVar = Var::make("p" + modeName + "_begin", Int());

  content->coordVar = Var::make(name, Int());
  content->segendVar = Var::make(modeName + "_segend", Int());
  content->validVar = Var::make("v" + modeName, Bool);
}

bool Iterator::isRoot() const {
  return !getParent().defined();
}

bool Iterator::isLeaf() const {
  return !getChild().defined();
}

const Iterator& Iterator::getParent() const {
  taco_iassert(defined());
  return content->parent;
}

const Iterator Iterator::getChild() const {
  taco_iassert(defined());
  return Iterator(content->child.lock());
}

void Iterator::setChild(const Iterator& iterator) const {
  taco_iassert(defined());
  content->child = iterator.content; 
}

IndexVar Iterator::getIndexVar() const {
  return content->indexVar;
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

bool Iterator::isDimensionIterator() const {
  return !content->mode.defined() && !content->tensor.defined();
}

bool Iterator::isModeIterator() const {
  return content->mode.defined();
}

bool Iterator::isFull() const {
  taco_iassert(defined());
  if (isDimensionIterator()) return true;
  return getMode().defined() && getMode().getModeFormat().isFull();
}

bool Iterator::isOrdered() const {
  taco_iassert(defined());
  if (isDimensionIterator()) return true;
  return getMode().defined() && getMode().getModeFormat().isOrdered();
}

bool Iterator::isUnique() const {
  taco_iassert(defined());
  if (isDimensionIterator()) return true;
  return getMode().defined() && getMode().getModeFormat().isUnique();
}

bool Iterator::isBranchless() const {
  taco_iassert(defined());
  if (isDimensionIterator()) return true;
  return getMode().defined() && getMode().getModeFormat().isBranchless();
}

bool Iterator::isCompact() const {
  taco_iassert(defined());
  if (isDimensionIterator()) return true;
  return getMode().defined() && getMode().getModeFormat().isCompact();
}

bool Iterator::hasCoordIter() const {
  taco_iassert(defined());
  if (isDimensionIterator()) return false;
  return getMode().defined() && getMode().getModeFormat().hasCoordValIter();
}

bool Iterator::hasPosIter() const {
  taco_iassert(defined());
  if (isDimensionIterator()) return false;
  return getMode().defined() && getMode().getModeFormat().hasCoordPosIter();
}

bool Iterator::hasLocate() const {
  taco_iassert(defined());
  if (isDimensionIterator()) return false;
  return getMode().defined() && getMode().getModeFormat().hasLocate();
}

bool Iterator::hasInsert() const {
  taco_iassert(defined());
  if (isDimensionIterator()) return false;
  return getMode().defined() && getMode().getModeFormat().hasInsert();
}

bool Iterator::hasAppend() const {
  taco_iassert(defined());
  if (isDimensionIterator()) return false;
  return getMode().defined() && getMode().getModeFormat().hasAppend();
}

ModeFunction Iterator::coordBounds(const std::vector<ir::Expr>& coords) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->coordIterBounds(coords, getMode());
}

ModeFunction Iterator::coordAccess(const std::vector<ir::Expr>& coords) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->coordIterAccess(getParent().getPosVar(),
                                                   coords, getMode());
}

ModeFunction Iterator::posBounds(const ir::Expr& parentPos) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->posIterBounds(parentPos, getMode());
}

ModeFunction Iterator::posAccess(const ir::Expr& pos, 
                                 const std::vector<ir::Expr>& coords) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->posIterAccess(pos, coords, getMode());
}

ModeFunction Iterator::locate(const std::vector<ir::Expr>& coords) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->locate(getParent().getPosVar(),
                                              coords, getMode());
}

Stmt Iterator::getInsertCoord(const Expr& p, const std::vector<Expr>& coords) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->getInsertCoord(p, coords, getMode());
}

Expr Iterator::getWidth() const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->getWidth(getMode());
}

Stmt Iterator::getInsertInitCoords(const Expr& pBegin, const Expr& pEnd) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->getInsertInitCoords(pBegin, pEnd,
                                                           getMode());
}

Stmt Iterator::getInsertInitLevel(const Expr& szPrev, const Expr& sz) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->getInsertInitLevel(szPrev, sz,
                                                          getMode());
}

Stmt Iterator::getInsertFinalizeLevel(const Expr& szPrev, const Expr& sz) const{
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->getInsertFinalizeLevel(szPrev, sz,
                                                              getMode());
}

Stmt Iterator::getAppendCoord(const Expr& p, const Expr& i) const {
  taco_iassert(defined() && content->mode.defined());
  return content->mode.getModeFormat().impl->getAppendCoord(p, i, content->mode);
}

Stmt Iterator::getAppendEdges(const Expr& pPrev, const Expr& pBegin, 
                              const Expr& pEnd) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->getAppendEdges(pPrev, pBegin, pEnd,
                                                      getMode());
}

Expr Iterator::getSize(const ir::Expr& szPrev) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->getSize(szPrev, getMode());
}

Stmt Iterator::getAppendInitEdges(const Expr& pPrevBegin, 
                                  const Expr& pPrevEnd) const {
  taco_iassert(defined() && content->mode.defined());
  return content->mode.getModeFormat().impl->getAppendInitEdges(pPrevBegin,
                                                              pPrevEnd,
                                                              content->mode);
}

Stmt Iterator::getAppendInitLevel(const Expr& szPrev, const Expr& sz) const {
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->getAppendInitLevel(szPrev, sz,
                                                          getMode());
}

Stmt Iterator::getAppendFinalizeLevel(const Expr& szPrev, const Expr& sz) const{
  taco_iassert(defined() && content->mode.defined());
  return getMode().getModeFormat().impl->getAppendFinalizeLevel(szPrev, sz,
                                                              getMode());
}

bool Iterator::defined() const {
  return content != nullptr;
}

bool operator==(const Iterator& a, const Iterator& b) {
  if (a.isDimensionIterator() && b.isDimensionIterator()) {
    return a.getIndexVar() == b.getIndexVar();
  }
  return a.content == b.content;
}

bool operator<(const Iterator& a, const Iterator& b) {
  if (a == b) return false;
  return a.content < b.content;
}

std::ostream& operator<<(std::ostream& os, const Iterator& iterator) {
  // Undefined iterator
  if (!iterator.defined()) {
    return os << "Iterator()";
  }
  // Dimension iterator
  if (iterator.isDimensionIterator()) {
    return os << "\u0394" << iterator.getIndexVar().getName();
  }
  return os << iterator.getTensor();
}


// class Iterators
struct Iterators::Content {
  map<ModeAccess,Iterator> levelIterators;
  map<Iterator,ModeAccess> modeAccesses;
  map<IndexVar,Iterator>   modeIterators;
};


Iterators::Iterators()
  : content(new Content)
{
}


static std::map<TensorVar, ir::Expr> createIRTensorVars(IndexStmt stmt)
{
  std::map<TensorVar, ir::Expr> tensorVars;

  // Create result and parameter variables
  vector<TensorVar> results = getResults(stmt);
  vector<TensorVar> arguments = getArguments(stmt);
  vector<TensorVar> temporaries = getTemporaries(stmt);

  // Convert tensor results, arguments and temporaries to IR variables
  map<TensorVar, Expr> resultVars;
  vector<Expr> resultsIR = createVars(results, &resultVars);
  tensorVars.insert(resultVars.begin(), resultVars.end());
  vector<Expr> argumentsIR = createVars(arguments, &tensorVars);
  vector<Expr> temporariesIR = createVars(temporaries, &tensorVars);

  return tensorVars;
}


Iterators::Iterators(IndexStmt stmt) : Iterators(stmt, createIRTensorVars(stmt))
{
}


Iterators::Iterators(IndexStmt stmt, const map<TensorVar, Expr>& tensorVars)
: Iterators()
{
  // Create dimension iteratorss
  match(stmt,
    function<void(const ForallNode*, Matcher*)>([&](auto n, auto m) {
      content->modeIterators.insert({n->indexVar, n->indexVar});
      m->match(n->stmt);
    })
  );

  // Create access iterators
  match(stmt,
    function<void(const AccessNode*)>([&](auto n) {
      taco_iassert(util::contains(tensorVars, n->tensorVar));
      Expr tensorIR = tensorVars.at(n->tensorVar);
      Format format = n->tensorVar.getFormat();
      createAccessIterators(Access(n), format, tensorIR);
    }),
    function<void(const AssignmentNode*, Matcher*)>([&](auto n, auto m) {
      m->match(n->rhs);
      m->match(n->lhs);
    })
  );

  // Reverse the levelITerators map for fast modeAccess lookup
  for (auto& iterator : content->levelIterators) {
    content->modeAccesses.insert({iterator.second, iterator.first});
  }
}


void
Iterators::createAccessIterators(Access access, Format format, Expr tensorIR)
{
  TensorVar tensorConcrete = access.getTensorVar();
  taco_iassert(tensorConcrete.getOrder() == format.getOrder())
      << tensorConcrete << ", Format" << format;
  Shape shape = tensorConcrete.getType().getShape();

  Iterator parent(tensorIR);
  content->levelIterators.insert({{access,0}, parent});

  int level = 1;
  ModeFormat parentModeType;
  for (ModeFormatPack modeTypePack : format.getModeFormatPacks()) {
    vector<Expr> arrays;
    taco_iassert(modeTypePack.getModeFormats().size() > 0);

    int modeNumber = format.getModeOrdering()[level-1];
    ModePack modePack(modeTypePack.getModeFormats().size(),
                      modeTypePack.getModeFormats()[0], tensorIR,
                      modeNumber, level);

    int pos = 0;
    for (auto& modeType : modeTypePack.getModeFormats()) {
      int modeNumber = format.getModeOrdering()[level-1];
      Dimension dim = shape.getDimension(modeNumber);
      IndexVar indexVar = access.getIndexVars()[modeNumber];
      Mode mode(tensorIR, dim, level, modeType, modePack, pos,
                parentModeType);

      string name = indexVar.getName() + tensorConcrete.getName();
      Iterator iterator(indexVar, tensorIR, mode, parent, name);
      content->levelIterators.insert({{access,modeNumber+1}, iterator});

      parent = iterator;
      parentModeType = modeType;
      pos++;
      level++;
    }
  }
}

Iterator Iterators::levelIterator(ModeAccess modeAccess) const
{
  taco_iassert(content != nullptr);
  taco_iassert(util::contains(content->levelIterators, modeAccess))
      << "Cannot find " << modeAccess << " in "
      << util::join(content->levelIterators);
  return content->levelIterators.at(modeAccess);
}


ModeAccess Iterators::modeAccess(Iterator iterator) const
{
  taco_iassert(content != nullptr);
  taco_iassert(util::contains(content->modeAccesses, iterator));
  return content->modeAccesses.at(iterator);
}


Iterator Iterators::modeIterator(IndexVar indexVar) const
{
  taco_iassert(content != nullptr);
  taco_iassert(util::contains(content->modeIterators, indexVar));
  return content->modeIterators.at(indexVar);
}


// Free functions
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
