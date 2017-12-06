#include "taco/expr/expr_nodes.h"

#include <set>
#include "taco/util/collections.h"

using namespace std;

namespace taco {

vector<TensorBase> getOperands(const IndexExpr& expr) {
  struct GetOperands : public ExprVisitor {
    using ExprVisitor::visit;
    set<TensorBase> inserted;
    vector<TensorBase> operands;
    void visit(const AccessNode* node) {
      auto tensor = node->tensor;
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

}
