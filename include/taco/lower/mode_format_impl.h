#ifndef TACO_MODE_FORMAT_IMPL_H
#define TACO_MODE_FORMAT_IMPL_H

#include <vector>
#include <initializer_list>
#include <memory>
#include <string>
#include <map>

#include "taco/format.h"
#include "taco/ir/ir.h"
#include "taco/lower/mode.h"

namespace taco {

class ModeFormatImpl;
class ModeFormatPack;
class ModePack;

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
  ir::Stmt compute() const;

  /// Retrieve the ith mode function result.
  ir::Expr operator[](size_t i) const;

  /// The number of results
  size_t numResults() const;

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
class ModeFormatImpl {
public:
  ModeFormatImpl(std::string name, bool isFull, bool isOrdered, bool isUnique, 
                 bool isBranchless, bool isCompact, bool hasCoordValIter, 
                 bool hasCoordPosIter, bool hasLocate, bool hasInsert, 
                 bool hasAppend);

  virtual ~ModeFormatImpl();

  /// Create a copy of the mode type with different properties.
  virtual ModeFormat copy(
      std::vector<ModeFormat::Property> properties) const = 0;


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
  virtual ModeFunction posIterAccess(ir::Expr pos, 
                                     std::vector<ir::Expr> coords,
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

  virtual ir::Expr getWidth(Mode mode) const;

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

  virtual ir::Expr getSize(ir::Expr parentSize, Mode mode) const;

  virtual ir::Stmt
  getAppendInitEdges(ir::Expr pPrevBegin, ir::Expr pPrevEnd, Mode mode) const;

  virtual ir::Stmt
  getAppendInitLevel(ir::Expr szPrev, ir::Expr sz, Mode mode) const;

  virtual ir::Stmt
  getAppendFinalizeLevel(ir::Expr szPrev, ir::Expr sz, Mode mode) const;
  /// @}

  /// Returns arrays associated with a tensor mode
  virtual std::vector<ir::Expr>
  getArrays(ir::Expr tensor, int mode, int level) const = 0;

  friend bool operator==(const ModeFormatImpl&, const ModeFormatImpl&);
  friend bool operator!=(const ModeFormatImpl&, const ModeFormatImpl&);

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

protected:
  /// Check if other mode format is identical. Can assume that this method will 
  /// always be called with an argument that is of the same class.
  virtual bool equals(const ModeFormatImpl& other) const;
};

static const int DEFAULT_ALLOC_SIZE = 1 << 20;

}
#endif

