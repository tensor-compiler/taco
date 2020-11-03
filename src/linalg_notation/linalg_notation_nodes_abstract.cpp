#include "taco/linalg_notation/linalg_notation_nodes_abstract.h"

#include "taco/linalg_notation/linalg_notation.h"
#include "taco/index_notation/schedule.h"
#include "taco/index_notation/transformations.h"

#include <tuple>

using namespace std;

namespace taco {

LinalgExprNode::LinalgExprNode(Datatype type)
  : dataType(type) {
}

Datatype LinalgExprNode::getDataType() const {
  return dataType;
}
}
