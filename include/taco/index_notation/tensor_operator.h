#ifndef TACO_OPS_H
#define TACO_OPS_H

#include <vector>
#include <functional>
#include <map>

#include "taco/ir/ir.h"
#include "taco/util/collections.h"
#include "taco/index_notation/properties.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_nodes.h"

namespace taco {

class Op {

using opImpl = TensorOpNode::opImpl;
using algebraImpl = TensorOpNode::algebraImpl;
using regionDefinition = TensorOpNode::regionDefinition;

public:
  Op(opImpl lowererFunc, algebraImpl algebraFunc, std::map<std::vector<int>, regionDefinition> specialDefinitions) :
          Op(lowererFunc, algebraFunc, Properties(), specialDefinitions) {}

  Op(opImpl lowererFunc, algebraImpl algebraFunc, Properties properties = Properties(),
          std::map<std::vector<int>, regionDefinition> specialDefinitions = {}) :
          lowererFunc(lowererFunc), algebraFunc(algebraFunc),
          properties(properties), regionDefinitions(specialDefinitions) {}

  template<typename... IndexExprs>
  TensorOp operator()(IndexExprs&&... exprs) {
    std::vector<IndexExpr> actualArgs{exprs...};
    IterationAlgebra nodeAlgebra = algebraFunc(actualArgs);
    Datatype returnType = inferReturnType(actualArgs);

    TensorOpNode* op = new TensorOpNode(actualArgs, lowererFunc, nodeAlgebra, properties,
                                        regionDefinitions, returnType);

    return TensorOp(op, util::uniqueName("Op"));
  }


private:
  opImpl lowererFunc;
  algebraImpl algebraFunc;
  Properties properties;
  std::map<std::vector<int>, regionDefinition> regionDefinitions;

  Datatype inferReturnType(const std::vector<IndexExpr>& inputs) {
    std::function<ir::Expr(IndexExpr)> getExprs = [](IndexExpr arg) { return ir::Var::make("t", arg.getDataType()); };
    std::vector<ir::Expr> exprs = util::map(inputs, getExprs);
    return lowererFunc(exprs).type();
  }

};

}
#endif //TACO_OPS_H

// Using vectors for interface to keep it consistent

// Can't use variadic functions since the lower function would need to be stored meaning methods in the compiler
// would have to be templated.
