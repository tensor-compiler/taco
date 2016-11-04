#ifndef TACO_STORAGE_ROOT_ITERATOR_H
#define TACO_STORAGE_ROOT_ITERATOR_H

#include "iterator.h"
#include "ir.h"

namespace taco {
namespace storage {

class RootIterator : public IteratorImpl {
public:
  RootIterator();
  virtual ~RootIterator() {};

  ir::Expr getPtrVar() const;
  ir::Expr getIdxVar() const;

  ir::Expr getIteratorVar() const;
  ir::Expr begin() const;
  ir::Expr end() const;

  ir::Stmt initDerivedVars() const;

  ir::Stmt storePtr() const;
  ir::Stmt storeIdx(ir::Expr idx) const;

  bool isRandomAccess() const;
};

}}
#endif
