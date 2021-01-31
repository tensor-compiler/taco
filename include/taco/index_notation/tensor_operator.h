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

class Func {
// TODO: RENAME
using OpImpl = CallNode::OpImpl;
using AlgebraImpl = CallNode::AlgebraImpl;

// TODO: Make this part of callNode and call. Add generateIterationAlgebra() and generateImplementation() functions
public:
  // Full construction
  Func(OpImpl lowererFunc, AlgebraImpl algebraFunc, std::vector<Property> properties,
       std::map<std::vector<int>, OpImpl> specialDefinitions = {});

  Func(std::string name, OpImpl lowererFunc, AlgebraImpl algebraFunc, std::vector<Property> properties,
       std::map<std::vector<int>, OpImpl> specialDefinitions = {});

  // Construct without specifying algebra
  Func(std::string name, OpImpl lowererFunc, std::vector<Property> properties,
       std::map<std::vector<int>, OpImpl> specialDefinitions  = {});

  Func(OpImpl lowererFunc, std::vector<Property> properties,
       std::map<std::vector<int>, OpImpl> specialDefinitions = {});

  // Construct without properties
  Func(std::string name, OpImpl lowererFunc, AlgebraImpl algebraFunc,
       std::map<std::vector<int>, OpImpl> specialDefinitions = {});

  Func(OpImpl lowererFunc, AlgebraImpl algebraFunc, std::map<std::vector<int>, OpImpl> specialDefinitions = {});

  // Construct without algebra or properties
  Func(std::string name, OpImpl lowererFunc, std::map<std::vector<int>, OpImpl> specialDefinitions = {});

  explicit Func(OpImpl lowererFunc, std::map<std::vector<int>, OpImpl> specialDefinitions = {});

  template<typename... IndexExprs>
  Call operator()(IndexExprs&&... exprs) {
    std::vector<IndexExpr> actualArgs{exprs...};

    IterationAlgebra nodeAlgebra = algebraFunc == nullptr? inferAlgFromProperties(actualArgs): algebraFunc(actualArgs);

    CallNode* op = new CallNode(name, actualArgs, lowererFunc, nodeAlgebra, properties,
                                regionDefinitions);

    return Call(op);
  }

private:
  std::string name;
  OpImpl lowererFunc;
  AlgebraImpl algebraFunc;
  std::vector<Property> properties;
  std::map<std::vector<int>, OpImpl> regionDefinitions;

  IterationAlgebra inferAlgFromProperties(const std::vector<IndexExpr>& exprs);

  // Constructs an algebra for iterating over the operator assuming the annihilator
  // of the expression is the input to this function.
  // Returns a pair where pair.first is the algebra for iteration and pair.second is
  // the number of subregions iterated by the algebra.
  IterationAlgebra constructAnnihilatorAlg(const std::vector<IndexExpr>& args, Annihilator annihilator);

  IterationAlgebra constructIdentityAlg(const std::vector<IndexExpr>& args, Identity identity);


  // Constructs an algebra that iterates over the entire space
  static IterationAlgebra constructDefaultAlgebra(const std::vector<IndexExpr>& exprs);
};

}
#endif //TACO_OPS_H


