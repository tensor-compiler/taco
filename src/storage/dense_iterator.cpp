#include "dense_iterator.h"

#include "util/strings.h"

using namespace taco::ir;

namespace taco {
namespace storage {

DenseIterator::DenseIterator(std::string name, const Expr& tensor, int level,
                             Iterator parent) {
  this->tensor = tensor;
  this->level = level;
  this->parentPtrVar = parent.getPtrVar();

  std::string indexVarName = name + util::toString(tensor);
  ptrVar = Var::make(indexVarName+"_ptr", typeOf<int>(), false);
  idxVar = Var::make(indexVarName, typeOf<int>(), false);
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
  return GetProperty::make(tensor, TensorProperty::Pointer, level);
}

Stmt DenseIterator::initDerivedVars() const {
  Expr ptrVal = Add::make(Mul::make(parentPtrVar, end()), getIdxVar());
  return VarAssign::make(getPtrVar(), ptrVal);
}

}}
