#include "root_iterator.h"

namespace taco {
namespace storage {

RootIterator::RootIterator() : zero(0) {
}

const ir::Expr& RootIterator::getIteratorVar() const {
  return zero;
}

const ir::Expr& RootIterator::getIndexVar() const {
  ierror << "The root iterator does not have an index var";
  return zero;
}

}}
