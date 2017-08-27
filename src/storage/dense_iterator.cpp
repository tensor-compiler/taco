#include "dense_iterator.h"

#include "taco/util/strings.h"

using namespace taco::ir;

namespace taco {
namespace storage {

DenseIterator::DenseIterator(std::string name, const Expr& tensor, int level,
                             size_t dimSize, Iterator previous)
      : IteratorImpl(previous, tensor) {
  this->tensor = tensor;
  this->level = level;

  std::string indexVarName = name + util::toString(tensor);
  ptrVar = Var::make("p" + util::toString(tensor) + std::to_string(level + 1),
                     Type(Type::Int));
  idxVar = Var::make(indexVarName, Type(Type::Int));

  this->dimSize = (int)dimSize;
}

bool DenseIterator::isDense() const {
  return true;
}

bool DenseIterator::isFixedRange() const {
  return true;
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
  if (isa<Literal>(dimSize) && to<Literal>(dimSize)->value <= 16) {
    return dimSize;
  }
  return getSizeArr();
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

ir::Stmt DenseIterator::initStorage(ir::Expr size) const {
  return Stmt();
}

ir::Stmt DenseIterator::resizePtrStorage(ir::Expr size) const {
  return Stmt();
}

ir::Stmt DenseIterator::resizeIdxStorage(ir::Expr size) const {
  return Stmt();
}

ir::Expr DenseIterator::getSizeArr() const {
  return GetProperty::make(tensor, TensorProperty::Dimensions, level);
}

}}
