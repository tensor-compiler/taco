#include "iterators.h"

#include <iostream>
#include <vector>
#include <map>

#include "taco/index_notation/index_notation.h"
#include "taco/format.h"
#include "taco/ir/ir.h"
#include "taco/error.h"
#include "taco/util/collections.h"

using namespace std;
using namespace taco::ir;
using taco::storage::Iterator;

namespace taco {
namespace lower {

// class Iterators
Iterators::Iterators() {
}

Iterators::Iterators(const IterationGraph& graph,
                     const map<TensorVar,ir::Expr>& tensorVariables) {
  // Create an iterator for each path step
  for (TensorPath path : util::combine(graph.getTensorPaths(),
                                       {graph.getResultTensorPath()})) {
    TensorVar tensorVar = path.getAccess().getTensorVar();
    Format format = tensorVar.getFormat();
    ir::Expr tensorVarExpr = tensorVariables.at(tensorVar);

    storage::Iterator parent = Iterator::makeRoot(tensorVarExpr);
    roots.insert({path, parent});

    ModeType prevModeType;
    modePacks.push_back(std::unique_ptr<ModePack>(new ModePack()));
    for (int i = 0, j = 0; i < (int)path.getSize(); ++i) {
      if (modePacks.back()->getSize() == 
          format.getModeTypePacks()[j].getModeTypes().size()) {
        modePacks.push_back(std::unique_ptr<ModePack>(new ModePack()));
        ++j;
      }

      std::string indexVarName = path.getVariables()[i].getName();
      ModeType modeType = format.getModeTypes()[i];
      size_t modeOrdering = format.getModeOrdering()[i];
      Dimension dim = tensorVar.getType().getShape().getDimension(modeOrdering);
      size_t pos = modePacks.back()->getSize();

      modePacks.back()->modes.emplace_back(tensorVarExpr, i, dim, 
                                           modePacks.back().get(), pos, 
                                           prevModeType);
      modePacks.back()->modeTypes.push_back(modeType);
      prevModeType = modeType;

      taco_iassert(path.getStep(i).getStep() == i);
      Iterator iterator = Iterator::make(path, indexVarName, tensorVarExpr, 
          modeType, &modePacks.back()->modes.back(), parent);
      iterators.insert({path.getStep(i), iterator});
      parent = iterator;
    }
  }
}

const storage::Iterator&
Iterators::operator[](const TensorPathStep& step) const {
  taco_iassert(util::contains(iterators, step)) <<
      "No iterator for step: " << step;
  return iterators.at(step);
}

vector<storage::Iterator>
Iterators::operator[](const vector<TensorPathStep>& steps) const {
  vector<storage::Iterator> iterators;
  for (auto& step : steps) {
    iterators.push_back((*this)[step]);
  }
  return iterators;
}

const storage::Iterator& Iterators::getRoot(const TensorPath& path) const {
  taco_iassert(util::contains(roots, path)) <<
      path << " does not have a root iterator";
  return roots.at(path);
}


// functions
std::vector<storage::Iterator>
getFullIterators(const std::vector<storage::Iterator>& iterators) {
  vector<storage::Iterator> fullIterators;
  for (auto& iterator : iterators) {
    if (iterator.defined() && iterator.isFull()) {
      fullIterators.push_back(iterator);
    }
  }
  return fullIterators;
}

//vector<storage::Iterator>
//getSequentialAccessIterators(const vector<storage::Iterator>& iterators) {
//  vector<storage::Iterator> sequentialAccessIterators;
//  for (auto& iterator : iterators) {
//    if (iterator.defined() && iterator.isSequentialAccess()) {
//      sequentialAccessIterators.push_back(iterator);
//    }
//  }
//  return sequentialAccessIterators;
//}
//
//vector<storage::Iterator>
//getRandomAccessIterators(const vector<storage::Iterator>& iterators) {
//  vector<storage::Iterator> randomAccessIterators;
//  for (auto& iterator : iterators) {
//    if (iterator.defined() && iterator.isRandomAccess()) {
//      randomAccessIterators.push_back(iterator);
//    }
//  }
//  return randomAccessIterators;
//}

vector<ir::Expr> getIdxVars(const vector<storage::Iterator>& iterators) {
  vector<ir::Expr> idxVars;
  for (auto& iterator : iterators) {
    idxVars.push_back(iterator.getIdxVar());
  }
  return idxVars;
}

}}
