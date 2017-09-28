#ifndef TACO_STORAGE_ITERATOR_H
#define TACO_STORAGE_ITERATOR_H

#include <memory>
#include <ostream>
#include <string>

#include "taco/ir/ir.h"
#include "taco/util/comparable.h"

namespace taco {
class TensorBase;

namespace ir {
class Stmt;
class Expr;
}

namespace storage {
class IteratorImpl;

/// A compile-time iterator over a tensor storage level. This class can be used
/// to generate the IR expressions for iterating over different level types.
class Iterator : public util::Comparable<Iterator> {
public:
  Iterator();

  static Iterator makeRoot(const ir::Expr& tensor);
  static Iterator make(std::string name, const ir::Expr& tensorVar,
                       size_t mode, ModeType modeType, size_t modeOrdering,
                       Iterator parent, const TensorBase& tensor);

  /// Get the parent of this iterator in its iterator list.
  const Iterator& getParent() const;

  /// Returns the tensor this iterator is iterating over.
  ir::Expr getTensor() const;

  /// Returns true if the iterator iterates over the entire tensor mode
  bool isDense() const;

  /// Returns true if the iterator iterates over ranges of fixed size.
  bool isFixedRange() const;

  /// Returns true if the iterator supports random access
  bool isRandomAccess() const;

  /// Returns true if the iterator supports sequential access
  bool isSequentialAccess() const;

  /// Returns the ptr variable for this iterator (e.g. `ja_ptr`). Ptr variables
  /// are used to index into the data at the next level (as well as the index
  /// arrays for formats such as sparse that have them).
  ir::Expr getPtrVar() const;

  /// Returns the index variable for this iterator (e.g. `ja`). Index variables
  /// are merged together using `min` in the emitted code to produce the loop
  /// index variable (e.g. `j`).
  ir::Expr getIdxVar() const;

  /// Returns the iterator variable. This is the variable that will iterate over
  /// the range [begin,end) with an increment of 1 in the emitted loop.
  ir::Expr getIteratorVar() const;

  /// Retrieves the expression that initializes the iterator variable before the
  /// loop starts executing.
  ir::Expr begin() const;

  /// Retrieves the expression that the iterator variable will be tested agains
  /// in the loop and that determines the end of the iterator.
  ir::Expr end() const;

  /// Returns a statement that initializes loop variables that are derived from
  /// the iterator variable.
  ir::Stmt initDerivedVar() const;

  /// Returns a statement that stores the ptr variable to the ptr index array.
  ir::Stmt storePtr() const;

  /// Returns a statement that stores `idx` to the idx index array.
  ir::Stmt storeIdx(ir::Expr idx) const;

  ir::Stmt initStorage(ir::Expr size) const;

  ir::Stmt resizePtrStorage(ir::Expr size) const;

  ir::Stmt resizeIdxStorage(ir::Expr size) const;

  /// Returns true if the iterator is defined, false otherwise.
  bool defined() const;

  friend bool operator==(const Iterator&, const Iterator&);
  friend bool operator<(const Iterator&, const Iterator&);
  friend std::ostream& operator<<(std::ostream&, const Iterator&);

private:
  std::shared_ptr<IteratorImpl> iterator;
};


/// Abstract class for iterators over different types of storage levels.
class IteratorImpl {
public:
  IteratorImpl(Iterator parent, ir::Expr tensor);
  virtual ~IteratorImpl();

  std::string getName() const;

  const Iterator& getParent() const;
  const ir::Expr& getTensor() const;

  virtual bool isDense() const                           = 0;
  virtual bool isFixedRange() const                      = 0;

  virtual bool isRandomAccess() const                    = 0;
  virtual bool isSequentialAccess() const                = 0;

  virtual ir::Expr getPtrVar() const                     = 0;
  virtual ir::Expr getIdxVar() const                     = 0;

  virtual ir::Expr getIteratorVar() const                = 0;
  virtual ir::Expr begin() const                         = 0;
  virtual ir::Expr end() const                           = 0;

  virtual ir::Stmt initDerivedVars() const               = 0;

  virtual ir::Stmt storeIdx(ir::Expr idx) const          = 0;
  virtual ir::Stmt storePtr() const                      = 0;

  virtual ir::Stmt initStorage(ir::Expr size) const      = 0;
  virtual ir::Stmt resizePtrStorage(ir::Expr size) const = 0;
  virtual ir::Stmt resizeIdxStorage(ir::Expr size) const = 0;

private:
  Iterator parent;
  ir::Expr tensor;
};

std::ostream& operator<<(std::ostream&, const IteratorImpl&);

}}
#endif
