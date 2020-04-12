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

public:
  // Full construction
  Op(opImpl lowererFunc, algebraImpl algebraFunc, std::vector<Property> properties,
     std::map<std::vector<int>, opImpl> specialDefinitions = {});

  Op(std::string name, opImpl lowererFunc, algebraImpl algebraFunc, std::vector<Property> properties,
     std::map<std::vector<int>, opImpl> specialDefinitions = {});

  // Construct without specifying algebra
  Op(std::string name, opImpl lowererFunc, std::vector<Property> properties,
     std::map<std::vector<int>, opImpl> specialDefinitions  = {});

  Op(opImpl lowererFunc, std::vector<Property> properties,
     std::map<std::vector<int>, opImpl> specialDefinitions = {});

  // Construct without properties
  Op(std::string name, opImpl lowererFunc, algebraImpl algebraFunc,
     std::map<std::vector<int>, opImpl> specialDefinitions = {});

  Op(opImpl lowererFunc, algebraImpl algebraFunc, std::map<std::vector<int>, opImpl> specialDefinitions = {});

  // Construct without algebra or properties
  Op(std::string name, opImpl lowererFunc, std::map<std::vector<int>, opImpl> specialDefinitions = {});

  explicit Op(opImpl lowererFunc, std::map<std::vector<int>, opImpl> specialDefinitions = {});

  template<typename... IndexExprs>
  TensorOp operator()(IndexExprs&&... exprs) {
    std::vector<IndexExpr> actualArgs{exprs...};

    IterationAlgebra nodeAlgebra = algebraFunc == nullptr? inferAlgFromProperties(actualArgs): algebraFunc(actualArgs);

    TensorOpNode* op = new TensorOpNode(name, actualArgs, lowererFunc, nodeAlgebra, properties,
                                        regionDefinitions);

    return TensorOp(op);
  }

private:
  std::string name;
  opImpl lowererFunc;
  algebraImpl algebraFunc;
  std::vector<Property> properties;
  std::map<std::vector<int>, opImpl> regionDefinitions;

  IterationAlgebra inferAlgFromProperties(const std::vector<IndexExpr>& exprs);

  // Constructs an algebra for iterating over the operator assuming the annihilator
  // of the expression is the input to this function.
  // Returns a pair where pair.first is the algebra for iteration and pair.second is
  // the number of subregions iterated by the algebra.
  std::pair<IterationAlgebra, int> constructAnnihilatorAlg(const std::vector<IndexExpr>& args,
                                                           Annihilator annihilator);

  // Constructs an algebra that iterates over the entire space
  static IterationAlgebra constructDefaultAlgebra(const std::vector<IndexExpr>& exprs);
};

}
#endif //TACO_OPS_H


