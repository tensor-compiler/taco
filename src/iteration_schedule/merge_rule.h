#ifndef TACO_MERGE_RULE_H
#define TACO_MERGE_RULE_H

#include <ostream>

#include "tensor_path.h"
#include "util/intrusive_ptr.h"

namespace taco {
namespace internal {
class Tensor;

/// Abstract superclass of the merge rules
struct MergeRuleNode : public util::Manageable<MergeRuleNode> {
  virtual ~MergeRuleNode();

protected:
  MergeRuleNode() = default;
};


/// A merge rule is a set-theoretic relationship between the iteration space
/// of tensor paths. They describe how to merge the tensor indices that are
/// incomming on an index variable to obtain the index variable's values.
class MergeRule : public util::IntrusivePtr<const MergeRuleNode> {
public:
  MergeRule();
  MergeRule(const MergeRuleNode*);

  /// Constructs a merge rule, given a tensor with a defined expression.
  static MergeRule make(const Tensor&);
};

std::ostream& operator<<(std::ostream&, const MergeRule&);


struct Path : public MergeRuleNode {
  Path(const TensorPath& path);
  static MergeRule make(const TensorPath& path);
  TensorPath path;
};


struct Union : public MergeRuleNode {
  static MergeRule make(MergeRule a, MergeRule b);
  MergeRule a, b;
};


struct Intersection : public MergeRuleNode {
  static MergeRule make(MergeRule a, MergeRule b);
  MergeRule a, b;
};

}}
#endif
