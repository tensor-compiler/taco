#include "taco/index_notation/index_notation_nodes_abstract.h"

#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/schedule.h"

using namespace std;

namespace taco {

// class ExprNode
IndexExprNode::IndexExprNode() : operatorSplits(new vector<OperatorSplit>) {
}

void IndexExprNode::splitOperator(IndexVar old, IndexVar left, IndexVar right) {
  operatorSplits->push_back(OperatorSplit(this, old, left, right));
}

IndexExprNode::IndexExprNode(DataType type)
    : operatorSplits(new vector<OperatorSplit>), dataType(type) {
}

DataType IndexExprNode::getDataType() const {
  return dataType;
}

const std::vector<OperatorSplit>& IndexExprNode::getOperatorSplits() const {
  return *operatorSplits;
}


// class TensorExprNode
IndexStmtNode::IndexStmtNode() {
}

IndexStmtNode::IndexStmtNode(Type type) : type (type) {
}

}
