#include "taco/index_notation/index_notation_nodes.h"

#include <set>
#include "taco/util/collections.h"

using namespace std;

namespace taco {

// class ReductionNode
ReductionNode::ReductionNode(IndexExpr op, IndexVar var, IndexExpr a)
    : IndexExprNode(a.getDataType()), op(op), var(var), a(a) {
  taco_iassert(isa<BinaryExprNode>(op.ptr));
}

}
