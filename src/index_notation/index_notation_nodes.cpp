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

// class CallNode
  CallNode::CallNode(std::string name, const std::vector<IndexExpr>& args, OpImpl defaultLowerFunc,
                     const IterationAlgebra &iterAlg, const std::vector<Property> &properties,
                     const std::map<std::vector<int>, OpImpl>& regionDefinitions)
          : CallNode(name, args, defaultLowerFunc, iterAlg, properties, regionDefinitions, definedIndices(args)){
  }

// class CallNode
CallNode::CallNode(std::string name, const std::vector<IndexExpr>& args, OpImpl defaultLowerFunc,
                   const IterationAlgebra &iterAlg, const std::vector<Property> &properties,
                   const std::map<std::vector<int>, OpImpl>& regionDefinitions,
                   const std::vector<int>& definedRegions)
    : IndexExprNode(inferReturnType(defaultLowerFunc, args)), name(name), args(args), defaultLowerFunc(defaultLowerFunc),
      iterAlg(applyDemorgan(iterAlg)), properties(properties), regionDefinitions(regionDefinitions),
      definedRegions(definedRegions) {

    taco_iassert(defaultLowerFunc != nullptr);
    for (const auto& pair: regionDefinitions) {
      taco_iassert(args.size() >= pair.first.size());
    }
}

// class ReductionNode
ReductionNode::ReductionNode(IndexExpr op, IndexVar var, IndexExpr a)
    : IndexExprNode(a.getDataType()), op(op), var(var), a(a) {
  taco_iassert(isa<BinaryExprNode>(op.ptr) || isa<CallNode>(op.ptr));
}

IndexVarNode::IndexVarNode(const std::string& name, const Datatype& type) 
    : IndexExprNode(type), content(new Content) {
  
  if (!type.isInt() && !type.isUInt()) {
    taco_not_supported_yet << ". IndexVars must be integral type.";
  }

  content->name = name;      
}

std::string IndexVarNode::getName() const {
  return content->name;
}

bool operator==(const IndexVarNode& a, const IndexVarNode& b) {
  return a.content->name == b.content->name;
}

bool operator<(const IndexVarNode& a, const IndexVarNode& b) {
  return a.content->name < b.content->name;
}

}
