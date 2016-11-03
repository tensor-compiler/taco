#include "sparse_iterator.h"

#include "util/strings.h"

using namespace taco::ir;

namespace taco {
namespace storage {

SparseIterator::SparseIterator(std::string name, const Expr& tensor, int level,
                               Iterator parent) {
  this->tensor = tensor;
  this->level = level;
  this->parentPtrVar = parent.getPtrVar();

  std::string idxVarName = name + util::toString(tensor);
  ptrVar = Var::make(idxVarName+"_ptr", typeOf<int>(), false);
  idxVar = Var::make(idxVarName, typeOf<int>(), false);
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
  Expr ptrArr = GetProperty::make(tensor, TensorProperty::Pointer, level);
  Expr ptrVal = Load::make(ptrArr, this->parentPtrVar);
  return ptrVal;
}

Expr SparseIterator::end() const {
  Expr ptrArr = GetProperty::make(tensor, TensorProperty::Pointer, level);
  Expr ptrVal = Load::make(ptrArr, Add::make(this->parentPtrVar, 1));
  return ptrVal;
}

Stmt SparseIterator::initDerivedVars() const {
  Expr idxArr = GetProperty::make(tensor, TensorProperty::Index, level);
  Expr idxVal = Load::make(idxArr, getPtrVar());
  return VarAssign::make(getIdxVar(), idxVal);
}

}}
