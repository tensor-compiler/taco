#include "merge_rule.h"

#include <vector>
#include <stack>

#include "expr.h"
#include "expr_visitor.h"
#include "expr_nodes.h"

using namespace std;

namespace taco {
namespace internal {

// class MergeRuleNode
MergeRuleNode::~MergeRuleNode() {
}


// class MergeRule
MergeRule::MergeRule(const MergeRuleNode* n)
    : util::IntrusivePtr<const MergeRuleNode>(n) {
}

MergeRule::MergeRule() : util::IntrusivePtr<const MergeRuleNode>() {
}

MergeRule MergeRule::make(const Tensor& tensor, const Var&,
                          const std::map<Expr,TensorPath>& tensorPaths) {
  struct ComputeMergeRule : public ExprVisitor {
    ComputeMergeRule(const std::map<Expr,TensorPath>& tensorPaths)
        : tensorPaths(tensorPaths) {}
    const std::map<Expr,TensorPath>& tensorPaths;
    stack<MergeRule> mergeRules;
    void visit(const ReadNode* op) {
//      MergeRule rule = Path(tensorPaths);
    }
  };
  ComputeMergeRule computeMergeRule(tensorPaths);
  tensor.getExpr().accept(&computeMergeRule);
  std::cout << computeMergeRule.mergeRules.top() << std::endl;

}

std::ostream& operator<<(std::ostream& os, const MergeRule& mergeRule) {
  return os << "merge rule";
}


// class Path
Path::Path(const TensorPath& path) : path(path) {
}

MergeRule Path::make(const TensorPath& path) {
  auto* node = new Path(path);
  return node;
}


// class Union
MergeRule Union::make(MergeRule a, MergeRule b) {
  auto* node = new Union;
  node->a = a;
  node->b = b;
  return node;
}


// class Intersection
MergeRule Intersection::make(MergeRule a, MergeRule b) {
  auto node = new Intersection;
  node->a = a;
  node->b = b;
  return node;
}

}}
