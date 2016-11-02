#include "sparse_iterator.h"

#include "util/strings.h"

using namespace taco::ir;

namespace taco {
namespace storage {

SparseIterator::SparseIterator(std::string name, const Expr& tensor) {
  this->tensor = tensor;

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
  // TODO
  return Expr();
}

Expr SparseIterator::end() const {
  // TODO
  return Expr();
}

Stmt SparseIterator::initDerivedVars() const {
  //TODO
  return Stmt();
}

}}
