#include "sparse_iterator.h"

#include "taco/util/strings.h"

using namespace std;
using namespace taco::ir;

namespace taco {
namespace storage {

SparseIterator::SparseIterator(std::string name, const Expr& tensor, int level,
                               Iterator previous)
    : IteratorImpl(previous, tensor) {
  this->tensor = tensor;
  this->level = level;

  std::string idxVarName = name + util::toString(tensor);
  ptrVar = Var::make("p" + util::toString(tensor) + std::to_string(level + 1),
                     Type(Type::Int));
  idxVar = Var::make(idxVarName, Type(Type::Int));
}

bool SparseIterator::isDense() const {
  return false;
}

bool SparseIterator::isFixedRange() const {
  return false;
}

bool SparseIterator::isRandomAccess() const {
  return false;
}

bool SparseIterator::isSequentialAccess() const {
  return true;
}

Expr SparseIterator::getPtrVar() const {
  return ptrVar;
}

Expr SparseIterator::getIdxVar() const {
  return idxVar;
}

Expr SparseIterator::getIteratorVar() const {
  return ptrVar;
}

Expr SparseIterator::begin() const {
  return Load::make(getPtrArr(), getParent().getPtrVar());
}

Expr SparseIterator::end() const {
  return Load::make(getPtrArr(), Add::make(getParent().getPtrVar(), 1));
}

Stmt SparseIterator::initDerivedVars() const {
  return VarAssign::make(getIdxVar(), Load::make(getIdxArr(), getPtrVar()),
                         true);
}

ir::Stmt SparseIterator::storePtr() const {
  return Store::make(getPtrArr(),
                     Add::make(getParent().getPtrVar(), 1), getPtrVar());
}

ir::Stmt SparseIterator::storeIdx(ir::Expr idx) const {
  return Store::make(getIdxArr(), getPtrVar(), idx);
}

ir::Expr SparseIterator::getPtrArr() const {
  string name = tensor.as<Var>()->name + to_string(level + 1) + "_pos";
  return GetProperty::make(tensor, TensorProperty::Indices, level, 0, name);
}

ir::Expr SparseIterator::getIdxArr() const {
  string name = tensor.as<Var>()->name + to_string(level + 1) + "_idx";
  return GetProperty::make(tensor, TensorProperty::Indices, level, 1, name);
}

ir::Stmt SparseIterator::initStorage(ir::Expr size) const {
  return Block::make({Allocate::make(getPtrArr(), size),
                      Allocate::make(getIdxArr(), size),
                      Store::make(getPtrArr(), 0, 0)});
}

ir::Stmt SparseIterator::resizePtrStorage(ir::Expr size) const {
  return Allocate::make(getPtrArr(), size, true);
}

ir::Stmt SparseIterator::resizeIdxStorage(ir::Expr size) const {
  return Allocate::make(getIdxArr(), size, true);
}

}}
