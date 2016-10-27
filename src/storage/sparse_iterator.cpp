#include "sparse_iterator.h"

#include "util/strings.h"

using namespace std;
using namespace taco::ir;

namespace taco {
namespace storage {

SparseIterator::SparseIterator(std::string name, const ir::Expr& tensor) {
  this->tensor = tensor;

  string indexVarName = name + util::toString(tensor);

  iteratorVar = Var::make(indexVarName+"_ptr", typeOf<int>());
  indexVar = Var::make(indexVarName, typeOf<int>());
}

const ir::Expr& SparseIterator::getIteratorVar() const {
  return iteratorVar;
}

const ir::Expr& SparseIterator::getIndexVar() const {
  return indexVar;
}

}}
