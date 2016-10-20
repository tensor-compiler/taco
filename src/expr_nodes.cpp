#include "expr_nodes.h"

#include <set>

#include "util/collections.h"

using namespace std;

namespace taco {
namespace internal {
vector<Tensor> getOperands(Expr expr) {
  struct GetOperands : public ExprVisitor {
    set<Tensor> inserted;
    vector<Tensor> operands;
    void visit(const ReadNode* node) {
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

}}
