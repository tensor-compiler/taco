#include "taco/linalg_notation/linalg_notation_nodes_abstract.h"

using namespace std;

namespace taco {

LinalgExprNode::LinalgExprNode(Datatype type)
  : dataType(type), order(0), isColVec(false) {
}

LinalgExprNode::LinalgExprNode(Datatype type, int order)
  : dataType(type), order(order) {
  if (order != 1)
    isColVec = false;
  else
    isColVec = true;
}

LinalgExprNode::LinalgExprNode(Datatype type, int order, bool isColVec)
  : dataType(type), order(order) {
  if (order != 1)
    this->isColVec = false;
  else
    this->isColVec = isColVec;
}

Datatype LinalgExprNode::getDataType() const {
  return dataType;
}

int LinalgExprNode::getOrder() const {
  return order;
}

bool LinalgExprNode::isColVector() const {
  return isColVec;
}

void LinalgExprNode::setColVector(bool val) {
  isColVec = val;
}

}
