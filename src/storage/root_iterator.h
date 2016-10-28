#ifndef TACO_STORAGE_ROOT_ITERATOR_H
#define TACO_STORAGE_ROOT_ITERATOR_H

#include "iterator.h"
#include "ir.h"

namespace taco {
namespace storage {

class RootIterator : public LevelIterator {
public:
  RootIterator();

  const ir::Expr& getIteratorVar() const;
  const ir::Expr& getIndexVar() const;

private:
  ir::Expr zero;
};

}}
#endif
