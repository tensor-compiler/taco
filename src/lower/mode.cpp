#include "taco/lower/mode.h"

#include "taco/lower/mode_format_impl.h"
#include "taco/ir/ir.h"
#include "taco/type.h"
#include "taco/util/strings.h"
#include "taco/util/collections.h"

using namespace std;

namespace taco {

// class Mode
struct Mode::Content {
  ir::Expr   tensor;            /// the tensor containing mode
  Dimension  size;              /// the size of the mode
  int        mode;              /// the location of mode in a mode hierarchy
  ModeFormat modeFormat;        /// the type of the mode

  ModePack   modePack;          /// the pack that contains the mode
  size_t     packLoc;           /// position within pack containing mode

  ModeFormat parentModeFormat;  /// type of previous mode in the tensor

  std::map<std::string, ir::Expr> vars;
};

Mode::Mode() : content(nullptr) {
}

Mode::Mode(ir::Expr tensor, Dimension size, int mode, ModeFormat modeFormat,
     ModePack modePack, size_t packLoc, ModeFormat parentModeFormat)
    : content(new Content) {
  taco_iassert(modeFormat.defined());
  content->tensor = tensor;
  content->size = size;
  content->mode = mode;
  content->modeFormat = modeFormat;
  content->modePack = modePack;
  content->packLoc = packLoc;
  content->parentModeFormat = parentModeFormat;
}

std::string Mode::getName() const {
  return util::toString(getTensorExpr()) + std::to_string(getLevel());
}

ir::Expr Mode::getTensorExpr() const {
  return content->tensor;
}

Dimension Mode::getSize() const {
  return content->size;
}

int Mode::getLevel() const {
  return content->mode;
}

ModeFormat Mode::getModeFormat() const {
  return content->modeFormat;
}

ModePack Mode::getModePack() const {
  return content->modePack;
}

size_t Mode::getPackLocation() const {
  return content->packLoc;
}

ModeFormat Mode::getParentModeType() const {
  return content->parentModeFormat;
}

ir::Expr Mode::getVar(std::string varName) const {
  taco_iassert(hasVar(varName));
  return content->vars.at(varName);
}

bool Mode::hasVar(std::string varName) const {
  return util::contains(content->vars, varName);
}

void Mode::addVar(std::string varName, ir::Expr var) {
  taco_iassert(ir::isa<ir::Var>(var));
  content->vars[varName] = var;
}

bool Mode::defined() const {
  return content != nullptr;
}

std::ostream& operator<<(std::ostream& os, const Mode& mode) {
  return os << mode.getName();
}


// class ModePack
struct ModePack::Content {
  size_t numModes = 0;
  vector<ir::Expr> arrays;
};

ModePack::ModePack() : content(new Content) {
}

ModePack::ModePack(size_t numModes, ModeFormat modeType, ir::Expr tensor,
                   int mode, int level) : ModePack() {
  content->numModes = numModes;
  content->arrays = modeType.impl->getArrays(tensor, mode, level);
}

size_t ModePack::getNumModes() const {
  return content->numModes;
}

ir::Expr ModePack::getArray(size_t i) const {
  return content->arrays[i];
}

}
