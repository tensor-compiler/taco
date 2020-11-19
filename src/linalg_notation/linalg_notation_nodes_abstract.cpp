#include "taco/linalg_notation/linalg_notation_nodes_abstract.h"

using namespace std;

namespace taco {

LinalgExprNode::LinalgExprNode(Datatype type)
  : dataType(type), order(0), isColVec(false), block(0) {
}

LinalgExprNode::LinalgExprNode(Datatype type, int order)
  : dataType(type), order(order), block(0) {
  if (order != 1)
    isColVec = false;
  else
    isColVec = true;
}

LinalgExprNode::LinalgExprNode(Datatype type, int order, bool isColVec)
  : dataType(type), order(order), block(0) {
  if (order != 1)
    this->isColVec = false;
  else
    this->isColVec = isColVec;
}

LinalgExprNode::LinalgExprNode(Datatype type, int order, bool isColVec, int block)
  : dataType(type), order(order), block(block) {
  if (block == 0 && order != 1)
    this->isColVec = false;
  else if (block != 0 && order != 2)
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

bool LinalgExprNode::isBlocked() const {
  return block != 0;
}

int LinalgExprNode::getBlock() const {
  return block;
}

void LinalgExprNode::setColVector(bool val) {
  isColVec = val;
}
}
