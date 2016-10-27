#ifndef TACO_STORAGE_DENSE_H
#define TACO_STORAGE_DENSE_H

#include <string>

#include "iterator.h"
#include "ir.h"

namespace taco {
namespace storage {

class DenseIterator : public LevelIterator {
public:
  DenseIterator(std::string name, const ir::Expr& tensor);

  const ir::Expr& getIteratorVar() const;
  const ir::Expr& getIndexVar() const;

private:
  ir::Expr iteratorVar;
  ir::Expr indexVar;

  ir::Expr tensor;
};

}}
#endif
