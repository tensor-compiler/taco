#include "taco/index_notation/tensor_operator.h"

namespace taco {

// Full construction
Func::Func(FuncBodyGenerator lowererFunc, FuncAlgebraGenerator algebraFunc, std::vector<Property> properties,
           std::map<std::vector<int>, FuncBodyGenerator> specialDefinitions)
       : name(util::uniqueName("Func")), lowererFunc(lowererFunc), algebraFunc(algebraFunc),
         properties(properties), regionDefinitions(specialDefinitions) {
}

Func::Func(std::string name, FuncBodyGenerator lowererFunc, FuncAlgebraGenerator algebraFunc, std::vector<Property> properties,
           std::map<std::vector<int>, FuncBodyGenerator> specialDefinitions)
       : name(name), lowererFunc(lowererFunc), algebraFunc(algebraFunc), properties(properties),
         regionDefinitions(specialDefinitions) {
}

// Construct without specifying algebra
Func::Func(std::string name, FuncBodyGenerator lowererFunc, std::vector<Property> properties,
           std::map<std::vector<int>, FuncBodyGenerator> specialDefinitions)
       : Func(name, lowererFunc, nullptr, properties, specialDefinitions) {
}

Func::Func(FuncBodyGenerator lowererFunc, std::vector<Property> properties,
           std::map<std::vector<int>, FuncBodyGenerator> specialDefinitions)
       : Func(util::uniqueName("Func"), lowererFunc, nullptr, properties, specialDefinitions) {
}

// Construct without properties
Func::Func(std::string name, FuncBodyGenerator lowererFunc, FuncAlgebraGenerator algebraFunc,
           std::map<std::vector<int>, FuncBodyGenerator> specialDefinitions)
       : Func(name, lowererFunc, algebraFunc, {}, specialDefinitions) {
}

Func::Func(FuncBodyGenerator lowererFunc, FuncAlgebraGenerator algebraFunc,
           std::map<std::vector<int>, FuncBodyGenerator> specialDefinitions) :
        Func(util::uniqueName("Func"), lowererFunc, algebraFunc, {}, specialDefinitions) {
}

// Construct without algebra or properties
Func::Func(std::string name, FuncBodyGenerator lowererFunc, std::map<std::vector<int>, FuncBodyGenerator> specialDefinitions)
       : Func(name, lowererFunc, nullptr, specialDefinitions) {
}

Func::Func(FuncBodyGenerator lowererFunc, std::map<std::vector<int>, FuncBodyGenerator> specialDefinitions)
       : Func(lowererFunc, nullptr, specialDefinitions) {
}

IterationAlgebra Func::inferAlgFromProperties(const std::vector<IndexExpr>& exprs) {
  if(properties.empty()) {
    return constructDefaultAlgebra(exprs);
  }

  // Start with smallest regions first. So we first check for annihilator and positional annihilator
  if(findProperty<Annihilator>(properties).defined()) {
    Annihilator annihilator = findProperty<Annihilator>(properties);
    IterationAlgebra alg = constructAnnihilatorAlg(exprs, annihilator);
    if(alg.defined()) {
      return alg;
    }
  }

  // Idempotence here ...

  if(findProperty<Identity>(properties).defined()) {
    Identity identity = findProperty<Identity>(properties);
    IterationAlgebra alg = constructIdentityAlg(exprs, identity);
    if(alg.defined()) {
      return alg;
    }
  }

  return constructDefaultAlgebra(exprs);
}

// Constructs an algebra that iterates over the entire space
IterationAlgebra Func::constructDefaultAlgebra(const std::vector<IndexExpr>& exprs) {
  if(exprs.empty()) return Region();

  IterationAlgebra tensorsRegions(exprs[0]);
  for(size_t i = 1; i < exprs.size(); ++i) {
    tensorsRegions = Union(tensorsRegions, exprs[i]);
  }

  IterationAlgebra background = Complement(tensorsRegions);
  return Union(tensorsRegions, background);
}

IterationAlgebra Func::constructAnnihilatorAlg(const std::vector<IndexExpr> &args, taco::Annihilator annihilator) {
  if(args.size () < 2) {
    return IterationAlgebra();
  }

  Literal annVal = annihilator.annihilator();
  std::vector<IndexExpr> toIntersect;

  if(annihilator.positions().empty()) {
    for(IndexExpr arg : args) {
      if(equals(inferFill(arg), annVal)) {
        toIntersect.push_back(arg);
      }
    }
  } else {
    for(size_t idx : annihilator.positions()) {
      if(equals(inferFill(args[idx]), annVal)) {
        toIntersect.push_back(args[idx]);
      }
    }
  }

  if(toIntersect.empty()) {
    return IterationAlgebra();
  }

  IterationAlgebra alg = toIntersect[0];
  for(size_t i = 1; i < toIntersect.size(); ++i) {
    alg = Intersect(alg, toIntersect[i]);
  }

  return alg;
}

IterationAlgebra Func::constructIdentityAlg(const std::vector<IndexExpr> &args, taco::Identity identity) {
  if(args.size() < 2) {
    return IterationAlgebra();
  }

  Literal idntyVal = identity.identity();

  if(identity.positions().empty()) {
    for(IndexExpr arg : args) {
      if(!equals(inferFill(arg), idntyVal)) {
        return IterationAlgebra();
      }
    }
  }

  IterationAlgebra alg(args[0]);
  for(size_t i = 1; i < args.size(); ++i) {
    alg = Union(alg, args[i]);
  }
  return alg;
}

}