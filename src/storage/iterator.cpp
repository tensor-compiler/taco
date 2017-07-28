#include "iterator.h"

#include "root_iterator.h"
#include "dense_iterator.h"
#include "sparse_iterator.h"
#include "fixed_iterator.h"

#include "taco/tensor.h"
#include "taco/expr.h"
#include "taco/ir/ir.h"
#include "taco/storage/storage.h"
#include "taco/storage/array.h"
#include "taco/storage/array_util.h"
#include "taco/util/strings.h"

using namespace std;

namespace taco {
namespace storage {

// class Iterator
Iterator::Iterator() : iterator(nullptr) {
}

Iterator Iterator::makeRoot(const ir::Expr& tensor) {
  Iterator iterator;
  iterator.iterator = std::make_shared<RootIterator>(tensor);
  return iterator;
}

Iterator Iterator::make(string name, const ir::Expr& tensorVar,
                        int dim, ModeType dimType, int dimOrder,
                        Iterator parent, const TensorBase& tensor) {
  Iterator iterator;

  switch (dimType) {
    case ModeType::Dense: {
      size_t dimSize = tensor.getDimension(dimOrder);
      iterator.iterator =
          std::make_shared<DenseIterator>(name, tensorVar, dim, dimSize,
                                          parent);
      break;
    }
    case ModeType::Sparse: {
      iterator.iterator =
          std::make_shared<SparseIterator>(name, tensorVar, dim, parent);
      break;
    }
    case ModeType::Fixed: {
      auto dimIndex = tensor.getStorage().getIndex().getModeIndex(dim);
      int fixedSize = getValue<int>(dimIndex.getIndexArray(0), 0);
      iterator.iterator =
          std::make_shared<FixedIterator>(name, tensorVar, dim, fixedSize,
                                          parent);
      break;
    }
  }
  taco_iassert(iterator.defined());
  return iterator;
}

const Iterator& Iterator::getParent() const {
  taco_iassert(defined());
  return iterator->getParent();
}

bool Iterator::isDense() const {
  taco_iassert(defined());
  return iterator->isDense();
}

bool Iterator::isFixedRange() const {
  taco_iassert(defined());
  return iterator->isFixedRange();
}

bool Iterator::isRandomAccess() const {
  taco_iassert(defined());
  return iterator->isRandomAccess();
}

bool Iterator::isSequentialAccess() const {
  taco_iassert(defined());
  return iterator->isSequentialAccess();
}

ir::Expr Iterator::getTensor() const {
  taco_iassert(defined());
  return iterator->getTensor();
}

ir::Expr Iterator::getPtrVar() const {
  taco_iassert(defined());
  return iterator->getPtrVar();
}

ir::Expr Iterator::getIdxVar() const {
  taco_iassert(defined());
  return iterator->getIdxVar();
}

ir::Expr Iterator::getIteratorVar() const {
  taco_iassert(defined());
  return iterator->getIteratorVar();
}

ir::Expr Iterator::begin() const {
  taco_iassert(defined());
  return iterator->begin();
}

ir::Expr Iterator::end() const {
  taco_iassert(defined());
  return iterator->end();
}

ir::Stmt Iterator::initDerivedVar() const {
  taco_iassert(defined());
  return iterator->initDerivedVars();
}

ir::Stmt Iterator::storePtr() const {
  taco_iassert(defined());
  return iterator->storePtr();
}

ir::Stmt Iterator::storeIdx(ir::Expr idx) const {
  taco_iassert(defined());
  return iterator->storeIdx(idx);
}

ir::Stmt Iterator::initStorage(ir::Expr size) const {
  taco_iassert(defined());
  return iterator->initStorage(size);
}

ir::Stmt Iterator::resizePtrStorage(ir::Expr size) const {
  taco_iassert(defined());
  return iterator->resizePtrStorage(size);
}

ir::Stmt Iterator::resizeIdxStorage(ir::Expr size) const {
  taco_iassert(defined());
  return iterator->resizeIdxStorage(size);
}

bool Iterator::defined() const {
  return iterator != nullptr;
}

bool operator==(const Iterator& a, const Iterator& b) {
  return a.iterator == b.iterator;
}

bool operator<(const Iterator& a, const Iterator& b) {
  return a.iterator < b.iterator;
}

std::ostream& operator<<(std::ostream& os, const Iterator& iterator) {
  if (!iterator.defined()) {
    return os << "Iterator()";
  }
  return os << *iterator.iterator;
}


// class IteratorImpl
IteratorImpl::IteratorImpl(Iterator parent, ir::Expr tensor) :
    parent(parent), tensor(tensor) {
}

IteratorImpl::~IteratorImpl() {
}

std::string IteratorImpl::getName() const {
  return util::toString(tensor);
}

const Iterator& IteratorImpl::getParent() const {
  return parent;
}

const ir::Expr& IteratorImpl::getTensor() const {
  return tensor;
}

std::ostream& operator<<(std::ostream& os, const IteratorImpl& iterator) {
  return os << iterator.getName();
}

}}
