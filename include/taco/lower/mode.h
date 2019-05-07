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

/// A mode of a tensor.
class Mode {
public:
  /// Construct an undefined mode.
  Mode();

  /// Construct a tensor mode.
  Mode(ir::Expr tensor, Dimension size, int mode, ModeFormat modeFormat,
       ModePack modePack, size_t packLoc, ModeFormat parentModeFormat);

  /// Retrieve the name of the tensor mode.
  std::string getName() const;

  /// Retrieve the tensor that contains the mode.
  ir::Expr getTensorExpr() const;

  /// Retrieve the size of the tensor mode.
  Dimension getSize() const;

  /// Retrieve the mode of this mode in its the mode hierarchy.  The first
  /// mode in a mode hierarchy is at mode 1, and mode 0 is the root mode.
  int getLevel() const;

  /// Retrieve the format of the mode.
  ModeFormat getModeFormat() const;

  /// Retrieve the mode pack that stores the mode.
  ModePack getModePack() const;

  /// Retrieve the location of the mode in its mode pack.
  size_t getPackLocation() const;

  /// Retrieve the mode type of the parent mode in the mode hierarchy.
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
  ModePack(size_t numModes, ModeFormat modeType, ir::Expr tensor, int mode, 
           int level);

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
