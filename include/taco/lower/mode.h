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

class ModeTypeImpl;
class ModeTypePack;
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

std::ostream& operator<<(std::ostream&, const Mode&);


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


/// Mode functions implement parts of mode capabilities, such as position
/// iteration and locate.  The lower machinery requests mode functions from
/// mode type implementations (`ModeTypeImpl`) and use these to generate code
/// to iterate over and assemble tensors.
class ModeFunction {
public:
  /// Construct an undefined mode function.
  ModeFunction();

  /// Construct a mode function.
  ModeFunction(ir::Stmt body, const std::vector<ir::Expr>& results);

  /// Retrieve the mode function's body where arguments are inlined.  The body
  /// may be undefined (when the result expression compute the mode function).
  ir::Stmt getBody() const;

  /// True if the mode function has a body.
  bool hasBody() const;

  /// Retrieve the mode function's result expressions.
  const std::vector<ir::Expr>& getResults() const;

  /// True if the mode function is defined.
  bool defined() const;

private:
  struct Content;
  std::shared_ptr<Content> content;
};

std::ostream& operator<<(std::ostream&, const ModeFunction&);


/// The abstract class to inherit from to add a new mode format to the system.
/// The mode type implementation can then be passed to the `ModeType`
/// constructor.
class ModeTypeImpl {
public:
  ModeTypeImpl() = delete;
  ModeTypeImpl(std::string name, bool isFull, bool isOrdered,
               bool isUnique, bool isBranchless, bool isCompact,
               bool hasCoordValIter, bool hasCoordPosIter, bool hasLocate,
               bool hasInsert, bool hasAppend);

  virtual ~ModeTypeImpl() {}

  /// Create a copy of the mode type with different properties.
  virtual ModeType copy(std::vector<ModeType::Property> properties) const = 0;


  /// The coordinate iteration capability's iterator function computes a range
  /// [result[0], result[1]) of coordinates to iterate over.
  /// `coord_iter_bounds(i_{1}, ..., i_{k−1}) -> begin_{k}, end_{k}`
  virtual ModeFunction coordIterBounds(std::vector<ir::Expr> parentCoords,
                                       Mode mode) const;

  /// The coordinate iteration capability's access function maps a coordinate
  /// iterator variable to a position (result[0]) and reports if a position
  /// could not be found (result[1]).
  /// `coord_iter_access(p_{k−1}, i_{1}, ..., i_{k}) -> p_{k}, found`
  virtual ModeFunction coordIterAccess(ir::Expr parentPos,
                                       std::vector<ir::Expr> coords,
                                       Mode mode) const;


  /// The position iteration capability's iterator function computes a range
  /// [result[0], result[1]) of positions to iterate over.
  /// `pos_iter_bounds(p_{k−1}) -> begin_{k}, end_{k}`
  virtual ModeFunction posIterBounds(ir::Expr parentPos, Mode mode) const;

  /// The position iteration capability's access function maps a position
  /// iterator variable to a coordinate (result[0]) and reports if a coordinate
  /// could not be found (result[1]).
  /// `pos_iter_access(p_{k}, i_{1}, ..., i_{k−1}) -> i_{k}, found`
  virtual ModeFunction posIterAccess(ir::Expr pos, std::vector<ir::Expr> coords,
                                     Mode mode) const;


  /// The locate capability locates the position of a coordinate (result[0])
  /// and reports if the coordinate could not be found (result[1]).
  /// `locate(p_{k−1}, i_{1}, ..., i_{k}) -> p_{k}, found`
  virtual ModeFunction locate(ir::Expr parentPos,
                              std::vector<ir::Expr> coords,
                              Mode mode) const;


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

