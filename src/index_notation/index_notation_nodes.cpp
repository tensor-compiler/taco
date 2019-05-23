#include "taco/index_notation/index_notation_nodes.h"
#include "taco/index_notation/intrinsic.h"

#include <set>
#include <functional>

#include "taco/util/collections.h"

using namespace std;

namespace taco {

template <typename T>
static std::vector<Datatype> getDataTypes(const std::vector<T> args) {
  std::function<Datatype(T)> getType = [](T arg) { return arg.getDataType(); };
  return util::map(args, getType);
}


// class CastNode
CastNode::CastNode(IndexExpr a, Datatype newType)
    : IndexExprNode(newType), a(a) {
}

// class CallIntrinsicNode
CallIntrinsicNode::CallIntrinsicNode(const std::shared_ptr<Intrinsic>& func, 
                                     const std::vector<IndexExpr>& args) 
    : IndexExprNode(func->inferReturnType(getDataTypes(args))),
      func(func), args(args) {
}


// class ReductionNode
ReductionNode::ReductionNode(IndexExpr op, IndexVar var, IndexExpr a)
    : IndexExprNode(a.getDataType()), op(op), var(var), a(a) {
  taco_iassert(isa<BinaryExprNode>(op.ptr));
}

}
