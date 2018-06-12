#include "taco/index_notation/index_notation_nodes_abstract.h"

#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/schedule.h"

#include <tuple>

using namespace std;

namespace taco {

// class ExprNode
IndexExprNode::IndexExprNode() : workspace(nullptr) {
}

IndexExprNode::IndexExprNode(DataType type)
    : dataType(type), workspace(nullptr) {
}

DataType IndexExprNode::getDataType() const {
  return dataType;
}

void IndexExprNode::setWorkspace(IndexVar i, IndexVar iw,
                                 TensorVar workspace)  const {
  this->workspace =
      make_shared<tuple<IndexVar,IndexVar,TensorVar>>(i,iw,workspace);
}

Workspace IndexExprNode::getWorkspace() const {
  if (workspace == nullptr) {
    return Workspace();
  }
  return Workspace(this, get<0>(*workspace), get<1>(*workspace),
                   get<2>(*workspace));
}


// class TensorExprNode
IndexStmtNode::IndexStmtNode() {
}

IndexStmtNode::IndexStmtNode(Type type) : type (type) {
}

}
