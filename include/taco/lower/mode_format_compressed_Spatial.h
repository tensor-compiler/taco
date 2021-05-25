#ifndef TACO_MODE_FORMAT_COMPRESSED_SPATIAL_H
#define TACO_MODE_FORMAT_COMPRESSED_SPATIAL_H

#include "taco/lower/mode_format_compressed.h"

namespace taco {

class CompressedModeFormatSpatial : public CompressedModeFormat {
public:

  CompressedModeFormatSpatial();
  CompressedModeFormatSpatial(bool isFull, bool isOrdered,
                       bool isUnique, bool isZeroless, long long allocSize = DEFAULT_ALLOC_SIZE);

  ir::Stmt getAppendCoord(ir::Expr pos, ir::Expr coord, 
                          Mode mode) const override;
};

}

#endif
