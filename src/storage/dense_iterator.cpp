#include "dense_iterator.h"

#include "util/strings.h"

using namespace taco::ir;

namespace taco {
namespace storage {

DenseIterator::DenseIterator(std::string name, const Expr& tensor) {
  this->tensor = tensor;

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
  // TODO
  return Expr();
}

Expr DenseIterator::end() const {
  // TODO
  return Expr();
}

Stmt DenseIterator::initDerivedVars() const {
  // TODO
  return Stmt();
}

}}
