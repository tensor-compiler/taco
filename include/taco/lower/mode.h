#ifndef TACO_MODE_H
#define TACO_MODE_H

#include <string>

#include "taco/format.h"

namespace taco {

class ModePack;
class Dimension;

namespace ir {
class Expr;
}

/// One of the modes of a tensor.
class Mode {
public:
  /// Construct an undefined mode.
  Mode();

  /// Construct a tensor mode.
  Mode(ir::Expr tensor, Dimension size, size_t level, ModeFormat modeFormat,
       ModePack modePack, size_t packLoc, ModeFormat parentModeFormat);

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
  ModeFormat getModeType() const;

  /// Retrieve the mode pack that stores the mode.
  ModePack getModePack() const;

  /// Retrieve the location of the mode in its mode pack.
  size_t getPackLocation() const;

  /// Retrieve the mode type of the parent level in the mode hierarchy.
  ModeFormat getParentModeType() const;

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
  ModePack(size_t numModes, ModeFormat modeType, ir::Expr tensor, size_t level);

  /// Returns number of tensor modes belonging to mode pack.
  size_t getNumModes() const;

  /// Returns arrays shared by tensor modes.
  ir::Expr getArray(size_t i) const;

private:
  struct Content;
  std::shared_ptr<Content> content;
};

}
#endif
