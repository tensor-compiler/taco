#ifndef TACO_STORAGE_SPARSE_H
#define TACO_STORAGE_SPARSE_H

#include <string>

#include "iterator.h"
#include "ir.h"

namespace taco {
namespace storage {

class SparseIterator : public LevelIterator {
public:
  SparseIterator(std::string name, const ir::Expr& tensor);

  const ir::Expr& getIteratorVar() const;
  const ir::Expr& getIndexVar() const;

private:
  ir::Expr iteratorVar;
  ir::Expr indexVar;

  ir::Expr tensor;
};

}}
#endif
