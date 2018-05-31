#ifndef TACO_STORAGE_ITERATOR_H
#define TACO_STORAGE_ITERATOR_H

#include <memory>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

#include "taco/ir/ir.h"
#include "taco/storage/mode_format.h"
#include "taco/util/comparable.h"
#include "lower/tensor_path.h"

namespace taco {
class Type;

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

  static Iterator makeRoot(const ir::Expr& tensorVar);
  static Iterator make(const lower::TensorPath& path, std::string indexVarName, 
                       const ir::Expr& tensorVar, ModeType modeType, Mode* mode, 
                       Iterator parent);

  /// Get the tensor path this iterator list iterates over.
  /// TODO: Remove this method and the path field.
  const lower::TensorPath& getTensorPath() const;

  const Iterator& getParent() const;
  
  /// Returns the tensor this iterator is iterating over.
  ir::Expr getTensor() const;

  const Mode& getMode() const;

  /// Returns the ptr variable for this iterator (e.g. `ja_ptr`). Ptr variables
  /// are used to index into the data at the next level (as well as the index
  /// arrays for formats such as sparse that have them).
  ir::Expr getPosVar() const;

  /// Returns the index variable for this iterator (e.g. `ja`). Index variables
  /// are merged together using `min` in the emitted code to produce the loop
  /// index variable (e.g. `j`).
  ir::Expr getIdxVar() const;

  ir::Expr getIteratorVar() const;

  ir::Expr getDerivedVar() const;

  ir::Expr getEndVar() const;

  ir::Expr getSegendVar() const;

  ir::Expr getValidVar() const;

  ir::Expr getBeginVar() const;

  bool isFull() const;
  bool isOrdered() const; 
  bool isUnique() const;
  bool isBranchless() const; 
  bool isCompact() const; 

  bool hasCoordValIter() const;
  bool hasCoordPosIter() const; 
  bool hasLocate() const;
  bool hasInsert() const;
  bool hasAppend() const;
  
  std::tuple<ir::Stmt,ir::Expr,ir::Expr> getCoordIter(
      const std::vector<ir::Expr>& i) const;
  std::tuple<ir::Stmt,ir::Expr,ir::Expr> getCoordAccess(const ir::Expr& pPrev, 
      const std::vector<ir::Expr>& i) const;
  
  std::tuple<ir::Stmt,ir::Expr,ir::Expr> getPosIter(
      const ir::Expr& pPrev) const;
  std::tuple<ir::Stmt,ir::Expr,ir::Expr> getPosAccess(const ir::Expr& p, 
      const std::vector<ir::Expr>& i) const;
  
  std::tuple<ir::Stmt,ir::Expr,ir::Expr> getLocate(const ir::Expr& pPrev, 
      const std::vector<ir::Expr>& i) const;

  ir::Stmt getInsertCoord(const ir::Expr& p, 
      const std::vector<ir::Expr>& i) const;
  ir::Expr getSize() const;
  ir::Stmt getInsertInitCoords(const ir::Expr& pBegin, 
      const ir::Expr& pEnd) const;
  ir::Stmt getInsertInitLevel(const ir::Expr& szPrev, const ir::Expr& sz) const;
  ir::Stmt getInsertFinalizeLevel(const ir::Expr& szPrev, 
      const ir::Expr& sz) const;
  
  ir::Stmt getAppendCoord(const ir::Expr& p, const ir::Expr& i) const; 
  ir::Stmt getAppendEdges(const ir::Expr& pPrev, const ir::Expr& pBegin, 
      const ir::Expr& pEnd) const;
  ir::Stmt getAppendInitEdges(const ir::Expr& pPrevBegin, 
      const ir::Expr& pPrevEnd) const;
  ir::Stmt getAppendInitLevel(const ir::Expr& szPrev, const ir::Expr& sz) const;
  ir::Stmt getAppendFinalizeLevel(const ir::Expr& szPrev, 
      const ir::Expr& sz) const;

  /// Returns true if the iterator is defined, false otherwise.
  bool defined() const;

  friend bool operator==(const Iterator&, const Iterator&);
  friend bool operator<(const Iterator&, const Iterator&);
  friend std::ostream& operator<<(std::ostream&, const Iterator&);

private:
  std::shared_ptr<IteratorImpl> iterator;
  lower::TensorPath path;
};


class IteratorImpl {
public:
  IteratorImpl(const ir::Expr& tensorVar);
  IteratorImpl(Iterator parent, std::string indexVarName, 
               const ir::Expr& tensorVar, ModeType modeType, Mode* mode);

  std::string getName() const;

  const Iterator& getParent() const;
  const ir::Expr& getTensor() const;
  const Mode& getMode() const;

  ir::Expr getIdxVar() const;
  ir::Expr getPosVar() const;
  ir::Expr getEndVar() const;
  ir::Expr getSegendVar() const;
  ir::Expr getValidVar() const;
  ir::Expr getBeginVar() const;

  bool isFull() const;
  bool isOrdered() const; 
  bool isUnique() const;
  bool isBranchless() const; 
  bool isCompact() const;

  bool hasCoordValIter() const;
  bool hasCoordPosIter() const; 
  bool hasLocate() const;
  bool hasInsert() const;
  bool hasAppend() const;
  
  std::tuple<ir::Stmt,ir::Expr,ir::Expr> getCoordIter(
      const std::vector<ir::Expr>& i) const;
  std::tuple<ir::Stmt,ir::Expr,ir::Expr> getCoordAccess(const ir::Expr& pPrev, 
      const std::vector<ir::Expr>& i) const;

  std::tuple<ir::Stmt,ir::Expr,ir::Expr> getPosIter(
      const ir::Expr& pPrev) const;
  std::tuple<ir::Stmt,ir::Expr,ir::Expr> getPosAccess(const ir::Expr& p, 
      const std::vector<ir::Expr>& i) const;

  std::tuple<ir::Stmt,ir::Expr,ir::Expr> getLocate(const ir::Expr& pPrev, 
      const std::vector<ir::Expr>& i) const;

  ir::Stmt getInsertCoord(const ir::Expr& p, 
      const std::vector<ir::Expr>& i) const;
  ir::Expr getSize() const;
  ir::Stmt getInsertInitCoords(const ir::Expr& pBegin, 
      const ir::Expr& pEnd) const;
  ir::Stmt getInsertInitLevel(const ir::Expr& szPrev, const ir::Expr& sz) const;
  ir::Stmt getInsertFinalizeLevel(const ir::Expr& szPrev, 
      const ir::Expr& sz) const;
  
  ir::Stmt getAppendCoord(const ir::Expr& p, const ir::Expr& i) const; 
  ir::Stmt getAppendEdges(const ir::Expr& pPrev, const ir::Expr& pBegin, 
      const ir::Expr& pEnd) const;
  ir::Stmt getAppendInitEdges(const ir::Expr& pPrevBegin, 
      const ir::Expr& pPrevEnd) const;
  ir::Stmt getAppendInitLevel(const ir::Expr& szPrev, const ir::Expr& sz) const;
  ir::Stmt getAppendFinalizeLevel(const ir::Expr& szPrev, 
      const ir::Expr& sz) const;

private:
  Iterator parent;
  ir::Expr tensorVar;
  ir::Expr posVar;
  ir::Expr idxVar;
  ir::Expr endVar;
  ir::Expr segendVar;
  ir::Expr validVar;
  ir::Expr beginVar;
  ModeType modeType;
  Mode*    mode;
};

std::ostream& operator<<(std::ostream&, const IteratorImpl&);

}}
#endif
