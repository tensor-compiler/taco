#ifndef TACO_STORAGE_ITERATOR_H
#define TACO_STORAGE_ITERATOR_H

#include <memory>
#include <ostream>
#include <string>

namespace taco {
class Expr;
class Level;

namespace internal {
class Tensor;
}

namespace ir {
class Expr;
}

namespace storage {
class LevelIterator;

/// A compile-time iterator over a tensor storage level. This class can be used
/// to generate the IR expressions for iterating over different level types.
class Iterator {
public:
  Iterator();

  static Iterator makeRoot();
  static Iterator make(std::string name, const ir::Expr& tensorVar,
                       int level, Level levelFormat);

  const ir::Expr& getIteratorVar() const;
  const ir::Expr& getIndexVar() const;

  bool defined() const;

private:
  std::shared_ptr<LevelIterator> iterator;
};

std::ostream& operator<<(std::ostream&, const Iterator&);


/// Abstract class for iterators over different types of storage levels.
class LevelIterator {
public:
  virtual const ir::Expr& getIteratorVar() const = 0;
  virtual const ir::Expr& getIndexVar() const = 0;
};

}}
#endif
