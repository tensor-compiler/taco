#include "merge_rule.h"

#include <vector>
#include <stack>

#include "expr.h"
#include "expr_visitor.h"
#include "expr_nodes.h"

using namespace std;

namespace taco {
namespace is {

// class MergeRuleNode
MergeRuleNode::~MergeRuleNode() {
}

std::ostream& operator<<(std::ostream& os, const MergeRuleNode& node) {
  class MergeRulePrinter : public MergeRuleVisitor {
  public:
    MergeRulePrinter(std::ostream& os) : os(os) {
    }
    std::ostream& os;
    void visit(const Path* rule) {
      os << rule->path.getTensor().getName();
    }
    void visit(const And* rule) {
      rule->a.accept(this);
      os << " and ";
      rule->b.accept(this);
    }
    void visit(const Or* rule) {
      rule->a.accept(this);
      os << " or ";
      rule->b.accept(this);
    }
  };
  MergeRulePrinter printer(os);
  node.accept(&printer);
  return os;
}


// class MergeRule
MergeRule::MergeRule(const MergeRuleNode* n)
    : util::IntrusivePtr<const MergeRuleNode>(n) {
}

MergeRule::MergeRule() : util::IntrusivePtr<const MergeRuleNode>() {
}

MergeRule MergeRule::make(const internal::Tensor& tensor, const Var& var,
                          const map<Expr,TensorPath>& tensorPaths) {

  struct ComputeMergeRule : public internal::ExprVisitor {
	using ExprVisitor::visit;
    ComputeMergeRule(const std::map<Expr,TensorPath>& tensorPaths)
        : tensorPaths(tensorPaths) {}
    const std::map<Expr,TensorPath>& tensorPaths;
    MergeRule mergeRule;

    void visit(const internal::ReadNode* op) {
      mergeRule = Path::make(tensorPaths.at(op));
    }

    void createOrRule(const internal::BinaryExprNode* node) {
      node->lhs.accept(this);
      MergeRule a = mergeRule;
      node->rhs.accept(this);
      MergeRule b = mergeRule;
      mergeRule = Or::make(a, b);
    }

    void createAndRule(const internal::BinaryExprNode* node) {
      node->lhs.accept(this);
      MergeRule a = mergeRule;
      node->rhs.accept(this);
      MergeRule b = mergeRule;
      mergeRule = And::make(a, b);;
    }

    void visit(const internal::AddNode* op) {
//      createOrRule(op);
    }

    void visit(const internal::SubNode* op) {
      createOrRule(op);
    }

    void visit(const internal::MulNode* op) {
//      createAndRule(op);
    }

    void visit(const internal::DivNode* op) {
      createAndRule(op);
    }
  };
  ComputeMergeRule computeMergeRule(tensorPaths);
  tensor.getExpr().accept(&computeMergeRule);
  return computeMergeRule.mergeRule;
}

std::vector<TensorPath> MergeRule::getPaths() const {
  struct GetPathsVisitor : public is::MergeRuleVisitor {
    using MergeRuleVisitor::visit;
    vector<is::TensorPath> paths;
    void visit(const is::Path* rule) {
      paths.push_back(rule->path);
    }
  };
  GetPathsVisitor getPathsVisitor;
  this->accept(&getPathsVisitor);
  return getPathsVisitor.paths;
}

void MergeRule::accept(MergeRuleVisitor* v) const {
  ptr->accept(v);
}

std::ostream& operator<<(std::ostream& os, const MergeRule& mergeRule) {
  return os << *(mergeRule.ptr);
}


// class Path
Path::Path(const TensorPath& path) : path(path) {
}

MergeRule Path::make(const TensorPath& path) {
  auto* node = new Path(path);
  return node;
}

void Path::accept(MergeRuleVisitor* v) const {
  v->visit(this);
}


// class And
MergeRule And::make(MergeRule a, MergeRule b) {
  auto node = new And;
  node->a = a;
  node->b = b;
  return node;
}

void And::accept(MergeRuleVisitor* v) const {
  v->visit(this);
}


// class Or
MergeRule Or::make(MergeRule a, MergeRule b) {
  auto* node = new Or;
  node->a = a;
  node->b = b;
  return node;
}

void Or::accept(MergeRuleVisitor* v) const {
  v->visit(this);
}


// class MergeRuleVisitor
MergeRuleVisitor::~MergeRuleVisitor() {
}

void MergeRuleVisitor::visit(const Path* rule) {
}

void MergeRuleVisitor::visit(const And* rule) {
  rule->a.accept(this);
  rule->b.accept(this);
}

void MergeRuleVisitor::visit(const Or* rule) {
  rule->a.accept(this);
  rule->b.accept(this);
}

}}
