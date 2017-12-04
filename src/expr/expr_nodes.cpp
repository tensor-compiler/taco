#include "taco/expr/expr_nodes.h"

#include <set>
#include "taco/util/collections.h"

using namespace std;

namespace taco {

vector<taco::TensorBase> getOperands(const IndexExpr& expr) {
  struct GetOperands : public ExprVisitor {
	using ExprVisitor::visit;
    set<TensorBase> inserted;
    vector<TensorBase> operands;
    void visit(const AccessNode* node) {
      if (!util::contains(inserted, node->tensor)) {
        inserted.insert(node->tensor);
        operands.push_back(node->tensor);
      }
    }
  };
  GetOperands getOperands;
  expr.accept(&getOperands);
  return getOperands.operands;
}

}
