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

namespace lower {
class IterationSchedule;
struct MergeRuleNode;
class MergeRuleVisitor;

/// A merge rule is a boolean expression that shows how to merge the incoming
/// paths on an index variable. A merge rule implements the set relationship
/// between the iteration space of incoming tensor paths as a set builder.
class MergeRule : public util::IntrusivePtr<const MergeRuleNode> {
public:
  MergeRule();
  MergeRule(const MergeRuleNode*);

  /// Constructs a merge rule for the index expression and index variable.
  static MergeRule make(const Expr& indexExpr, const Var& indexVar,
                        const IterationSchedule& schedule);

  /// Returns the operand tensor path steps merged by this rule.
  std::vector<TensorPathStep> getSteps() const;

  void accept(MergeRuleVisitor*) const;
};

std::ostream& operator<<(std::ostream&, const MergeRule&);


/// Abstract superclass of the merge rules
struct MergeRuleNode : public util::Manageable<MergeRuleNode> {
  virtual ~MergeRuleNode();
  virtual void accept(MergeRuleVisitor*) const = 0;

protected:
  MergeRuleNode() = default;
};

std::ostream& operator<<(std::ostream&, const MergeRuleNode&);


/// The atoms of a merge rule is a step of a tensor path
struct Step : public MergeRuleNode {
  static MergeRule make(const TensorPathStep& step);

  virtual void accept(MergeRuleVisitor*) const;

  TensorPathStep step;
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


/// Visits merge rules
class MergeRuleVisitor {
public:
  virtual ~MergeRuleVisitor();
  virtual void visit(const Step* rule);
  virtual void visit(const And* rule);
  virtual void visit(const Or* rule);
};


/// Simplifies `rule` by taking advantage of dense iteration spaces being
/// super-sets of every other iteration space. Merge rules involving a dense
/// index D are simplified as follows
/// A and D -> A
/// A or  D -> D
MergeRule simplify(const MergeRule& rule);

}}
#endif
