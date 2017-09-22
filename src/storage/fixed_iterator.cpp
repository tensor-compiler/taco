#include "fixed_iterator.h"

#include "taco/util/strings.h"

using namespace taco::ir;

namespace taco {
namespace storage {

FixedIterator::FixedIterator(std::string name, const Expr& tensor, int level,
                             size_t fixedSize, Iterator previous)
    : IteratorImpl(previous, tensor) {
  this->tensor = tensor;
  this->level = level;

  std::string idxVarName = name + util::toString(tensor);
  ptrVar = Var::make(util::toString(tensor) + std::to_string(level) + "_ptr",
                     Type(Type::Int));
  idxVar = Var::make(idxVarName,Type(Type::Int));

  this->fixedSize = (int)fixedSize;
}

bool FixedIterator::isDense() const {
  return false;
}

bool FixedIterator::isFixedRange() const {
  return false;
}

bool FixedIterator::isRandomAccess() const {
  return false;
}

bool FixedIterator::isSequentialAccess() const {
  return true;
}

Expr FixedIterator::getPtrVar() const {
  return ptrVar;
}

Expr FixedIterator::getIdxVar() const {
  return idxVar;
}

Expr FixedIterator::getIteratorVar() const {
  return ptrVar;
}

Expr FixedIterator::begin() const {
  return 0;
}

Expr FixedIterator::end() const {
  return fixedSize;
}

Stmt FixedIterator::initDerivedVars() const {
  Expr ptrVal = Add::make(Mul::make(getParent().getPtrVar(), end()),
                          getIdxVar());
  return VarAssign::make(getIdxVar(), ptrVal);
}

ir::Stmt FixedIterator::storePtr() const {
  return Stmt();
}

ir::Stmt FixedIterator::storeIdx(ir::Expr idx) const {
  return Store::make(getIdxArr(), getPtrVar(), idx);
}

ir::Expr FixedIterator::getPtrArr() const {
  return GetProperty::make(tensor, TensorProperty::Dimension, level);
}

ir::Expr FixedIterator::getIdxArr() const {
  return GetProperty::make(tensor, TensorProperty::Dimension, level);
}

ir::Stmt FixedIterator::initStorage(ir::Expr size) const {
  return Block::make({Allocate::make(getPtrArr(), 1),
                      Allocate::make(getIdxArr(), size)});
}

ir::Stmt FixedIterator::resizePtrStorage(ir::Expr size) const {
  return Stmt();
}

ir::Stmt FixedIterator::resizeIdxStorage(ir::Expr size) const {
  return Allocate::make(getIdxArr(), size, true);
}

}}
