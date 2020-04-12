#include "taco/index_notation/tensor_operator.h"

namespace taco {

// Full construction
Op::Op(opImpl lowererFunc, algebraImpl algebraFunc, std::vector<Property> properties,
       std::map<std::vector<int>, opImpl> specialDefinitions)
       : name(util::uniqueName("Op")), lowererFunc(lowererFunc), algebraFunc(algebraFunc),
         properties(properties), regionDefinitions(specialDefinitions) {
}

Op::Op(std::string name, opImpl lowererFunc, algebraImpl algebraFunc, std::vector<Property> properties,
       std::map<std::vector<int>, opImpl> specialDefinitions)
       : name(name), lowererFunc(lowererFunc), algebraFunc(algebraFunc), properties(properties),
         regionDefinitions(specialDefinitions) {
}

// Construct without specifying algebra
Op::Op(std::string name, opImpl lowererFunc, std::vector<Property> properties,
       std::map<std::vector<int>, opImpl> specialDefinitions)
       : Op(name, lowererFunc, nullptr, properties, specialDefinitions) {
}

Op::Op(opImpl lowererFunc, std::vector<Property> properties,
       std::map<std::vector<int>, opImpl> specialDefinitions)
       : Op(util::uniqueName("Op"), lowererFunc, nullptr, properties, specialDefinitions) {
}

// Construct without properties
Op::Op(std::string name, opImpl lowererFunc, algebraImpl algebraFunc,
       std::map<std::vector<int>, opImpl> specialDefinitions)
       : Op(name, lowererFunc, algebraFunc, {}, specialDefinitions) {
}

Op::Op(opImpl lowererFunc, algebraImpl algebraFunc,
       std::map<std::vector<int>, opImpl> specialDefinitions) :
        Op(util::uniqueName("Op"), lowererFunc, algebraFunc, {}, specialDefinitions) {
}

// Construct without algebra or properties
Op::Op(std::string name, opImpl lowererFunc, std::map<std::vector<int>, opImpl> specialDefinitions)
       : Op(name, lowererFunc, nullptr, specialDefinitions) {
}

Op::Op(opImpl lowererFunc, std::map<std::vector<int>, opImpl> specialDefinitions)
       : Op(lowererFunc, nullptr, specialDefinitions) {
}

IterationAlgebra Op::inferAlgFromProperties(const std::vector<IndexExpr>& exprs) {
  if(properties.empty()) {
    return constructDefaultAlgebra(exprs);
  }

  // Start with smallest regions first. So we first check for annihilator and positional annihilator
  if(findProperty<Annihilator>(properties).defined()) {
    Literal annihilator = findProperty<Annihilator>(properties).annihilator();

  }

  return {};
}

// Constructs an algebra that iterates over the entire space
IterationAlgebra Op::constructDefaultAlgebra(const std::vector<IndexExpr>& exprs) {
  if(exprs.empty()) return Region();

  IterationAlgebra tensorsRegions(exprs[0]);
  for(size_t i = 1; i < exprs.size(); ++i) {
    tensorsRegions = Union(tensorsRegions, exprs[i]);
  }

  IterationAlgebra background = Complement(tensorsRegions);
  return Union(tensorsRegions, background);
}

std::pair<IterationAlgebra, int> Op::constructAnnihilatorAlg(const std::vector<IndexExpr> &args,
                                                             taco::Annihilator annihilator) {
  taco_iassert(args.size() > 1) << "Annihilator must be applied to operand with at least two arguments";

}

}