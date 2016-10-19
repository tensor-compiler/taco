#ifndef TACO_MERGE_RULE_H
#define TACO_MERGE_RULE_H

#include <ostream>
#include <map>

#include "tensor_path.h"
#include "util/intrusive_ptr.h"

namespace taco {
class Expr;

namespace internal {
class Tensor;
}

namespace is {
class MergeRuleVisitor;

/// Abstract superclass of the merge rules
struct MergeRuleNode : public util::Manageable<MergeRuleNode> {
  virtual ~MergeRuleNode();
  virtual void accept(MergeRuleVisitor*) const = 0;
protected:
  MergeRuleNode() = default;
};

std::ostream& operator<<(std::ostream&, const MergeRuleNode&);


/// A merge rule is a set-theoretic relationship between the iteration space
/// of tensor paths. They describe how to merge the tensor indices that are
/// incomming on an index variable to obtain the index variable's values.

/// A merge rule is a boolean expression that shows how to merge the incoming
/// paths on an index variable. A merge rule implements the set relationship
/// between the iteration space of incoming tensor paths as a set builder.
class MergeRule : public util::IntrusivePtr<const MergeRuleNode> {
public:
  MergeRule();
  MergeRule(const MergeRuleNode*);

  /// Constructs a merge rule, given a tensor with a defined expression.
  static MergeRule make(const internal::Tensor&, const Var&,
                        const std::map<Expr,TensorPath>&);
  void accept(MergeRuleVisitor*) const;
};

std::ostream& operator<<(std::ostream&, const MergeRule&);


/// The atoms of a merge rule are tensor paths
struct Path : public MergeRuleNode {
  Path(const TensorPath& path);
  static MergeRule make(const TensorPath& path);
  virtual void accept(MergeRuleVisitor*) const;
  TensorPath path;
};

/// And merge rules implements intersection relationships between sparse
/// iteration spaces
struct And : public MergeRuleNode {
  static MergeRule make(MergeRule a, MergeRule b);
  virtual void accept(MergeRuleVisitor*) const;
  MergeRule a, b;
};

/// Or merge rules implements union relationships between sparse iteration
/// spaces
struct Or : public MergeRuleNode {
  static MergeRule make(MergeRule a, MergeRule b);
  virtual void accept(MergeRuleVisitor*) const;
  MergeRule a, b;
};

class MergeRuleVisitor {
public:
  virtual ~MergeRuleVisitor();
  virtual void visit(const Path* rule);
  virtual void visit(const And* rule);
  virtual void visit(const Or* rule);
};

}}
#endif
