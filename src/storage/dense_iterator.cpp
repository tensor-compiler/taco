#include "dense_iterator.h"

#include "util/strings.h"

using namespace taco::ir;

namespace taco {
namespace storage {

DenseIterator::DenseIterator(std::string name, const Expr& tensor, int level,
                             size_t dimSize, Iterator previous)
      : IteratorImpl(previous) {
  this->tensor = tensor;
  this->level = level;

  std::string indexVarName = name + util::toString(tensor);
  ptrVar = Var::make(util::toString(tensor) + std::to_string(level+1)+"_ptr",
                     typeOf<int>(), false);
  idxVar = Var::make(indexVarName, typeOf<int>(), false);

  this->dimSize = (int)dimSize;
}

bool DenseIterator::isRandomAccess() const {
  return true;
}

bool DenseIterator::isSequentialAccess() const {
  // TODO: Change to true
  return false;
}

Expr DenseIterator::getPtrVar() const {
  return ptrVar;
}

Expr DenseIterator::getIdxVar() const {
  return idxVar;
}

Expr DenseIterator::getIteratorVar() const {
  return idxVar;
}

Expr DenseIterator::begin() const {
  return 0;
}

Expr DenseIterator::end() const {
  return dimSize;
}

Stmt DenseIterator::initDerivedVars() const {
  Expr ptrVal = Add::make(Mul::make(getParent().getPtrVar(), end()),
                          getIdxVar());
  return VarAssign::make(getPtrVar(), ptrVal);
}

ir::Stmt DenseIterator::storePtr() const {
  return Stmt();
}

ir::Stmt DenseIterator::storeIdx(ir::Expr idx) const {
  return Stmt();
}

ir::Stmt DenseIterator::resizePtrStorage(ir::Expr size) const {
  return Stmt();
}

ir::Stmt DenseIterator::resizeIdxStorage(ir::Expr size) const {
  return Stmt();
}

}}
