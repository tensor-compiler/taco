#ifndef TACO_MODE_TYPE_H
#define TACO_MODE_TYPE_H

#include <vector>
#include <initializer_list>
#include <memory>
#include <string>
#include <map>

#include "taco/format.h"
#include "taco/ir/ir.h"
#include "taco/util/strings.h"

namespace taco {

class IteratorImpl;
class ModeTypeImpl;
class ModeTypePack;
struct ModePack;

namespace old {
class Iterators;
}


struct Mode {
  Mode(const ir::Expr tensor, const size_t mode, const Dimension size, 
       const ModePack* const pack, const size_t pos, 
       const ModeType prevModeType);

  const ir::Expr        tensor;        // tensor containing mode
  const size_t          mode;          // identifier for mode
  const Dimension       size;          // size of mode

  const ModePack* const pack;          // reference to pack containing mode
  const size_t          pos;           // position within pack containing mode
  const ModeType        prevModeType;  // type of previous mode in containing tensor

  /// Returns a string that identifies the tensor mode
  std::string getName() const;

  ir::Expr getVar(const std::string varName) const;
  bool     hasVar(const std::string varName) const;
  void     addVar(const std::string varName, ir::Expr var);

private:
  // Stores temporary variables that may be needed to access or modify a mode
  std::map<std::string, ir::Expr> vars;
};


/// A mode pack consists of tensor modes that share the same physical arrays 
/// (e.g., modes of an array-of-structs COO tensor).
struct ModePack {
  /// Returns number of tensor modes belonging to mode pack.
  size_t getSize() const;

  /// Returns arrays shared by tensor modes.
  ir::Expr getArray(size_t idx) const;

private:
  std::vector<Mode> modes;
  std::vector<ModeType> modeTypes;

  friend class old::Iterators;
};

class ModeTypeImpl {
public:
  ModeTypeImpl() = delete;
  ModeTypeImpl(std::string formatName, bool isFull, bool isOrdered,
               bool isUnique, bool isBranchless, bool isCompact,
               bool hasCoordValIter, bool hasCoordPosIter, bool hasLocate,
               bool hasInsert, bool hasAppend);

  virtual ~ModeTypeImpl() {}

  /// Instantiates a variant of the mode type with differently configured 
  /// properties
  virtual ModeType copy(
      const std::vector<ModeType::Property>& properties) const = 0;

  /// Return code for level functions that implement coordinate value iteration
  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr> getCoordIter(
      const std::vector<ir::Expr>& i, Mode& mode) const;
  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr> getCoordAccess(
      const ir::Expr& pPrev, const std::vector<ir::Expr>& i, Mode& mode) const;
  
  /// Return code for level functions that implement coordinate position  
  /// iteration
  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr> getPosIter(
      const ir::Expr& pPrev, Mode& mode) const;
  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr> getPosAccess(const ir::Expr& p, 
      const std::vector<ir::Expr>& i, Mode& mode) const;
  
  /// Returns code for level function that implements locate capability
  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr> getLocate(
      const ir::Expr& pPrev, const std::vector<ir::Expr>& i, Mode& mode) const;

  /// Return code for level functions that implement insert capabilitiy
  virtual ir::Stmt getInsertCoord(const ir::Expr& p, 
      const std::vector<ir::Expr>& i, Mode& mode) const;
  virtual ir::Expr getSize(Mode& mode) const;
  virtual ir::Stmt getInsertInitCoords(const ir::Expr& pBegin, 
      const ir::Expr& pEnd, Mode& mode) const;
  virtual ir::Stmt getInsertInitLevel(const ir::Expr& szPrev, 
      const ir::Expr& sz, Mode& mode) const;
  virtual ir::Stmt getInsertFinalizeLevel(const ir::Expr& szPrev, 
      const ir::Expr& sz, Mode& mode) const;
  
  /// Return code for level functions that implement append capabilitiy
  virtual ir::Stmt getAppendCoord(const ir::Expr& p, const ir::Expr& i, 
      Mode& mode) const; 
  virtual ir::Stmt getAppendEdges(const ir::Expr& pPrev, const ir::Expr& pBegin, 
      const ir::Expr& pEnd, Mode& mode) const;
  virtual ir::Stmt getAppendInitEdges(const ir::Expr& pPrevBegin, 
      const ir::Expr& pPrevEnd, Mode& mode) const;
  virtual ir::Stmt getAppendInitLevel(const ir::Expr& szPrev, 
      const ir::Expr& sz, Mode& mode) const;
  virtual ir::Stmt getAppendFinalizeLevel(const ir::Expr& szPrev, 
      const ir::Expr& sz, Mode& mode) const;

  /// Returns arrays associated with a tensor mode
  virtual ir::Expr getArray(size_t idx, const Mode& mode) const = 0;

  const std::string formatName;

  const bool isFull;
  const bool isOrdered;
  const bool isUnique;
  const bool isBranchless;
  const bool isCompact;

  const bool hasCoordValIter;
  const bool hasCoordPosIter;
  const bool hasLocate;
  const bool hasInsert;
  const bool hasAppend;
};

}
#endif

