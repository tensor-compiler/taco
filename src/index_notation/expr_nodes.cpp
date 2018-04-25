#include "taco/index_notation/expr_nodes.h"

#include <set>
#include "taco/util/collections.h"

using namespace std;

namespace taco {

vector<TensorVar> getOperands(const IndexExpr& expr) {
  struct GetOperands : public ExprVisitor {
    using ExprVisitor::visit;
    set<TensorVar> inserted;
    vector<TensorVar> operands;
    void visit(const AccessNode* node) {
      TensorVar tensor = node->tensorVar;
      if (!util::contains(inserted, tensor)) {
        inserted.insert(tensor);
        operands.push_back(tensor);
      }
    }
  };
  GetOperands getOperands;
  expr.accept(&getOperands);
  return getOperands.operands;
}

// class ReductionNode
ReductionNode::ReductionNode(IndexExpr op, IndexVar var, IndexExpr a)
    : ExprNode(a.getDataType()), op(op), var(var), a(a) {
  taco_iassert(isa<BinaryExprNode>(op));
}

}
