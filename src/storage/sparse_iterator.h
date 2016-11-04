#ifndef TACO_STORAGE_SPARSE_H
#define TACO_STORAGE_SPARSE_H

#include <string>

#include "iterator.h"
#include "ir.h"

namespace taco {
namespace storage {

class SparseIterator : public IteratorImpl {
public:
  SparseIterator(std::string name, const ir::Expr& tensor, int level,
                 Iterator parent);
  virtual ~SparseIterator() {};

  ir::Expr getPtrVar() const;
  ir::Expr getIdxVar() const;

  ir::Expr getIteratorVar() const;
  ir::Expr begin() const;
  ir::Expr end() const;

  ir::Stmt initDerivedVars() const;

  ir::Stmt storePtr() const;
  ir::Stmt storeIdx(ir::Expr idx) const;

  bool isRandomAccess() const;

private:
  ir::Expr tensor;
  int level;

  ir::Expr parentPtrVar;

  ir::Expr ptrVar;
  ir::Expr idxVar;

  ir::Expr getPtrArr() const;
  ir::Expr getIdxArr() const;
};

}}
#endif
