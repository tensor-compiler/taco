#ifndef TACO_STORAGE_DENSE_H
#define TACO_STORAGE_DENSE_H

#include <string>

#include "iterator.h"
#include "ir.h"

namespace taco {
namespace storage {

class DenseIterator : public IteratorImpl {
public:
  DenseIterator(std::string name, const ir::Expr& tensor);

  ir::Expr getPtrVar() const;
  ir::Expr getIdxVar() const;

  ir::Expr getIteratorVar() const;
  ir::Expr begin() const;
  ir::Expr end() const;

  ir::Stmt initDerivedVars() const;

private:
  ir::Expr ptrVar;
  ir::Expr idxVar;

  ir::Expr tensor;
};

}}
#endif
