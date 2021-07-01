#include "taco/lower/mode_format_compressed_Spatial.h"
#include "taco/spatial.h"
#include "ir/ir_generators.h"
#include "taco/ir/simplify.h"
#include "taco/util/strings.h"

using namespace std;
using namespace taco::ir;

namespace taco {

CompressedModeFormatSpatial::CompressedModeFormatSpatial() :
    CompressedModeFormatSpatial(false, true, true, false) {
}

CompressedModeFormatSpatial::CompressedModeFormatSpatial(bool isFull, bool isOrdered,
                                           bool isUnique, bool isZeroless, 
                                           long long allocSize) :
  CompressedModeFormat(isFull, isOrdered, isUnique, isZeroless, allocSize) {
}

Stmt CompressedModeFormatSpatial::getAppendCoord(Expr p, Expr i, Mode mode) const {
  taco_iassert(mode.getPackLocation() == 0);

  Expr idxArray = getCoordArray(mode.getModePack());
  Expr stride = (int)mode.getModePack().getNumModes();
  // FIXME: [Spatial] make sure memory location isn't hardcoded
  Stmt storeIdx = Store::make(idxArray, ir::Mul::make(p, stride), i, MemoryLocation::SpatialFIFO, MemoryLocation::SpatialReg);

  return storeIdx;
}

Stmt CompressedModeFormatSpatial::getAppendEdges(Expr pPrev, Expr pBegin, Expr pEnd,
                                          Mode mode) const {
  Expr posArray = getPosArray(mode.getModePack());
  ModeFormat parentModeType = mode.getParentModeType();
  Expr edges = pEnd;
  cout << "GetAppendEdges Spatial: " << edges << endl;
  //Expr edges = (!parentModeType.defined() || parentModeType.hasAppend())
  //             ? pEnd : ir::Sub::make(pEnd, pBegin);
  return Store::make(posArray, ir::Add::make(pPrev, 1), edges, MemoryLocation::SpatialSparseDRAM, MemoryLocation::SpatialReg);
}

}
