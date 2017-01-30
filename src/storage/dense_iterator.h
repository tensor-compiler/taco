#ifndef TACO_STORAGE_DENSE_H
#define TACO_STORAGE_DENSE_H

#include <string>

#include "iterator.h"
#include "ir.h"

namespace taco {
namespace storage {

class DenseIterator : public IteratorImpl {
public:
  DenseIterator(std::string name, const ir::Expr& tensor, int level,
                Iterator parent, size_t dimSize);
  virtual ~DenseIterator() {};

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

  ir::Stmt resizePtrStorage(ir::Expr size) const;
  ir::Stmt resizeIdxStorage(ir::Expr size) const;

private:
  ir::Expr tensor;
  int level;

  ir::Expr parentPtrVar;

  ir::Expr ptrVar;
  ir::Expr idxVar;

  ir::Expr dimSize;
};

}}
#endif
