#ifndef TACO_MODE_TYPE_H
#define TACO_MODE_TYPE_H

#include <vector>
#include <initializer_list>
#include <memory>
#include <string>
#include <map>

#include "taco/format.h"
#include "taco/ir/ir.h"

namespace taco {

class IteratorImpl;
class ModeTypeImpl;
class ModeTypePack;
class ModePack;
class ModePack;

namespace old {
class Iterators;
}


/// One of the modes of a tensor.
class Mode {
public:
  /// Construct an undefined mode.
  Mode();

  /// Construct a tensor mode.
  Mode(ir::Expr tensor, Dimension size, size_t level, ModeType modeType,
       ModePack modePack, size_t packLoc, ModeType parentModeType);

  /// Retrieve the name of the tensor mode.
  std::string getName() const;

  /// Retrieve the tensor that contains the mode.
  ir::Expr getTensorExpr() const;

  /// Retrieve the size of the tensor mode.
  Dimension getSize() const;

  /// Retrieve the level of this mode in its the mode hierarchy.  The first
  /// mode in a mode hierarchy is at level 1, and level 0 is the root level.
  size_t getLevel() const;

  /// Retrieve the mode's type.
  ModeType getModeType() const;

  /// Retrieve the mode pack that stores the mode.
  ModePack getModePack() const;

  /// Retrieve the location of the mode in its mode pack.
  size_t getPackLocation() const;

  /// Retrieve the mode type of the parent level in the mode hierarchy.
  ModeType getParentModeType() const;

  /// Store temporary variables that may be needed to access or modify a mode
  /// @{
  ir::Expr getVar(std::string varName) const;
  bool     hasVar(std::string varName) const;
  void     addVar(std::string varName, ir::Expr var);
  /// @}

  /// Check whether the mode is defined.
  bool defined() const;

private:
  struct Content;
  std::shared_ptr<Content> content;
  friend class ModePack;
};


/// A ModePack is a set of physical arrays, that can be used by one mode or
/// shared by multiple modes (e.g., modes of an array-of-structs COO tensor).
class ModePack {
public:
  ModePack();
  ModePack(size_t numModes, ModeType modeType, ir::Expr tensor, size_t level);

  /// Returns number of tensor modes belonging to mode pack.
  size_t getNumModes() const;

  /// Returns arrays shared by tensor modes.
  ir::Expr getArray(size_t i) const;

private:
  struct Content;
  std::shared_ptr<Content> content;
};


class ModeTypeImpl {
public:
  ModeTypeImpl() = delete;
  ModeTypeImpl(std::string name, bool isFull, bool isOrdered,
               bool isUnique, bool isBranchless, bool isCompact,
               bool hasCoordValIter, bool hasCoordPosIter, bool hasLocate,
               bool hasInsert, bool hasAppend);

  virtual ~ModeTypeImpl() {}

  /// Instantiates a variant of the mode type with differently configured 
  /// properties
  virtual ModeType
  copy(const std::vector<ModeType::Property>& properties) const = 0;


  /// Level functions that implement coordinate value iteration.
  /// @{
  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr>
  getCoordIter(const std::vector<ir::Expr>& i, Mode mode) const;

  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr>
  getCoordAccess(ir::Expr pPrev, const std::vector<ir::Expr>& i,
                 Mode mode) const;
  /// @}


  /// Level functions that implement coordinate position iteration.
  /// @{
  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr>
  getPosIter(ir::Expr pPrev, Mode mode) const;

  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr>
  getPosAccess(ir::Expr p, const std::vector<ir::Expr>& i, Mode mode) const;
  /// @}


  /// Level function that implements locate capability.
  virtual std::tuple<ir::Stmt,ir::Expr,ir::Expr>
  getLocate(ir::Expr pPrev, const std::vector<ir::Expr>& i, Mode mode) const;


  /// Level functions that implement insert capabilitiy.
  /// @{
  virtual ir::Stmt
  getInsertCoord(ir::Expr p, const std::vector<ir::Expr>& i, Mode mode) const;

  virtual ir::Expr getSize(Mode mode) const;

  virtual ir::Stmt
  getInsertInitCoords(ir::Expr pBegin, ir::Expr pEnd, Mode mode) const;

  virtual ir::Stmt
  getInsertInitLevel(ir::Expr szPrev, ir::Expr sz, Mode mode) const;

  virtual ir::Stmt
  getInsertFinalizeLevel(ir::Expr szPrev, ir::Expr sz, Mode mode) const;
  /// @}

  
  /// Level functions that implement append capabilitiy.
  /// @{
  virtual ir::Stmt
  getAppendCoord(ir::Expr p, ir::Expr i, Mode mode) const;

  virtual ir::Stmt
  getAppendEdges(ir::Expr pPrev, ir::Expr pBegin, ir::Expr pEnd,
                 Mode mode) const;

  virtual ir::Stmt
  getAppendInitEdges(ir::Expr pPrevBegin, ir::Expr pPrevEnd, Mode mode) const;

  virtual ir::Stmt
  getAppendInitLevel(ir::Expr szPrev, ir::Expr sz, Mode mode) const;

  virtual ir::Stmt
  getAppendFinalizeLevel(ir::Expr szPrev, ir::Expr sz, Mode mode) const;
  /// @}

  /// Returns one of the arrays associated with a tensor mode
  virtual ir::Expr getArray(size_t idx, const Mode mode) const = 0;

  /// Returns arrays associated with a tensor mode
  virtual std::vector<ir::Expr>
  getArrays(ir::Expr tensor, size_t level) const = 0;

  const std::string name;

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

