#include "taco/index_notation/expr_node.h"

#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/schedule.h"

using namespace std;

namespace taco {

// class ExprNode
ExprNode::ExprNode() : operatorSplits(new vector<OperatorSplit>) {
}

void ExprNode::splitOperator(IndexVar old, IndexVar left, IndexVar right) {
  operatorSplits->push_back(OperatorSplit(this, old, left, right));
}

ExprNode::ExprNode(DataType type)
    : operatorSplits(new vector<OperatorSplit>), dataType(type) {
}

DataType ExprNode::getDataType() const {
  return dataType;
}

const std::vector<OperatorSplit>& ExprNode::getOperatorSplits() const {
  return *operatorSplits;
}

}
