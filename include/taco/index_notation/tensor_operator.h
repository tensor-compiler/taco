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

  // Full construction
  Op(opImpl lowererFunc, algebraImpl algebraFunc, std::vector<Property> properties = {},
     std::map<std::vector<int>, regionDefinition> specialDefinitions = {}) :
          name(util::uniqueName("Op")), lowererFunc(lowererFunc), algebraFunc(algebraFunc),
          properties(properties), regionDefinitions(specialDefinitions) {}

  Op(std::string name, opImpl lowererFunc, algebraImpl algebraFunc, std::vector<Property> properties = {},
     std::map<std::vector<int>, regionDefinition> specialDefinitions = {}) :
          name(name), lowererFunc(lowererFunc), algebraFunc(algebraFunc),
          properties(properties), regionDefinitions(specialDefinitions) {}

  // Construct without specifying algebra
  Op(std::string name, opImpl lowererFunc, std::vector<Property> properties,
     std::map<std::vector<int>, regionDefinition> specialDefinitions  = {}) :
          Op(name, lowererFunc, nullptr, properties, specialDefinitions) {}

  Op(opImpl lowererFunc, std::vector<Property> properties,
     std::map<std::vector<int>, regionDefinition> specialDefinitions = {}) :
          Op(util::uniqueName("Op"), lowererFunc, nullptr, properties, specialDefinitions) {}

  // Construct without algebra or properties
  Op(std::string name, opImpl lowererFunc) : Op(name, lowererFunc, nullptr) {}

  explicit Op(opImpl lowererFunc) : Op(lowererFunc, nullptr) {}


  template<typename... IndexExprs>
  TensorOp operator()(IndexExprs&&... exprs) {
    std::vector<IndexExpr> actualArgs{exprs...};

    IterationAlgebra nodeAlgebra = algebraFunc != nullptr? algebraFunc(actualArgs): inferAlgFromProperties(actualArgs);
    Datatype returnType = inferReturnType(actualArgs);

    TensorOpNode* op = new TensorOpNode(actualArgs, lowererFunc, nodeAlgebra, properties,
                                        regionDefinitions, returnType);

    return TensorOp(op, name);
  }


private:
  std::string name;
  opImpl lowererFunc;
  algebraImpl algebraFunc;
  std::vector<Property> properties;
  std::map<std::vector<int>, regionDefinition> regionDefinitions;

  Datatype inferReturnType(const std::vector<IndexExpr>& inputs) {
    std::function<ir::Expr(IndexExpr)> getExprs = [](IndexExpr arg) { return ir::Var::make("t", arg.getDataType()); };
    std::vector<ir::Expr> exprs = util::map(inputs, getExprs);
    return lowererFunc(exprs).type();
  }

  IterationAlgebra inferAlgFromProperties(const std::vector<IndexExpr>& exprs) {
    if(properties.empty()) {
      return constructDefaultAlgebra(exprs);
    }
    return {};
  }

  IterationAlgebra constructDefaultAlgebra(const std::vector<IndexExpr>& exprs) {
    return {};
  }

};

}
#endif //TACO_OPS_H

// Using vectors for interface to keep it consistent

// Can't use variadic functions since the lower function would need to be stored meaning methods in the compiler
// would have to be templated.
