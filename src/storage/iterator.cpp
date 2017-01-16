#include "iterator.h"

#include "expr.h"
#include "internal_tensor.h"
#include "ir.h"

#include "root_iterator.h"
#include "dense_iterator.h"
#include "sparse_iterator.h"

using namespace std;

namespace taco {
namespace storage {

// class Iterator
Iterator::Iterator() {
}

Iterator Iterator::makeRoot() {
  Iterator iterator;
  iterator.iterator = std::make_shared<RootIterator>();
  return iterator;
}

Iterator Iterator::make(string name, const ir::Expr& tensorVar,
                        int level, Level levelFormat, Iterator parent,
                        const internal::Tensor& tensor) {
  Iterator iterator;

  // TODO: Remove
  switch (levelFormat.getType()) {
    case LevelType::Dense: {
      size_t dimSize = tensor.getDimensions()[levelFormat.getDimension()];
      iterator.iterator =
          std::make_shared<DenseIterator>(name, tensorVar, level, parent,
                                          dimSize);
      break;
    }
    case LevelType::Sparse:
      iterator.iterator =
          std::make_shared<SparseIterator>(name, tensorVar, level, parent);
      break;
    case LevelType::Fixed:
      break;
  }
  iassert(iterator.defined());
  return iterator;
}

ir::Expr Iterator::getPtrVar() const {
  return iterator->getPtrVar();
}

ir::Expr Iterator::getIdxVar() const {
  return iterator->getIdxVar();
}

ir::Expr Iterator::getIteratorVar() const {
  return iterator->getIteratorVar();
}

ir::Expr Iterator::begin() const {
  return iterator->begin();
}

ir::Expr Iterator::end() const {
  return iterator->end();
}

ir::Stmt Iterator::initDerivedVar() const {
  return iterator->initDerivedVars();
}

ir::Stmt Iterator::storePtr() const {
  return iterator->storePtr();
}

ir::Stmt Iterator::storeIdx(ir::Expr idx) const {
  return iterator->storeIdx(idx);
}

ir::Stmt Iterator::resizePtrStorage(ir::Expr size) const {
  return iterator->resizePtrStorage(size);
}

ir::Stmt Iterator::resizeIdxStorage(ir::Expr size) const {
  return iterator->resizeIdxStorage(size);
}

bool Iterator::isRandomAccess() const {
  return iterator->isRandomAccess();
}

bool Iterator::defined() const {
  return iterator != nullptr;
}

std::ostream& operator<<(std::ostream& os, const Iterator& iterator) {
  return os << iterator.getPtrVar();
}


}}
