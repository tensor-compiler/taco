#ifndef TACO_STORAGE_ROOT_ITERATOR_H
#define TACO_STORAGE_ROOT_ITERATOR_H

#include "iterator.h"
#include "taco/ir/ir.h"

namespace taco {
namespace storage {

class RootIterator : public IteratorImpl {
public:
  RootIterator(const ir::Expr& tensor);
  virtual ~RootIterator() {};

  bool isDense() const;
  bool isFixedRange() const;

  bool isRandomAccess() const;
  bool isSequentialAccess() const;

  ir::Expr getPtrVar() const;
  ir::Expr getIdxVar() const;

  ir::Expr getIteratorVar() const;
  ir::Expr begin() const;
  ir::Expr end() const;

  ir::Stmt initDerivedVars() const;

  ir::Stmt storePtr() const;
  ir::Stmt storeIdx(ir::Expr idx) const;

  ir::Stmt initStorage(ir::Expr size) const;
  ir::Stmt resizePtrStorage(ir::Expr size) const;
  ir::Stmt resizeIdxStorage(ir::Expr size) const;
};

}}
#endif
