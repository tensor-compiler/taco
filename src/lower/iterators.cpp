#include "iterators.h"

#include <iostream>

#include "taco/expr/expr.h"
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
  for (auto& path : graph.getTensorPaths()) {
    TensorVar tensorVar = path.getTensor();
    Format format = tensorVar.getFormat();
    ir::Expr tensorVarExpr = tensorVariables.at(tensorVar);

    storage::Iterator parent = Iterator::makeRoot(tensorVarExpr);
    roots.insert({path, parent});

    for (int i=0; i < (int)path.getSize(); ++i) {
      string name = path.getVariables()[i].getName();

      Iterator iterator = Iterator::make(name, tensorVarExpr, i,
                                         format.getModeTypes()[i],
                                         format.getModeOrdering()[i], parent,
                                         tensorVar.getType());
      taco_iassert(path.getStep(i).getStep() == i);
      iterators.insert({path.getStep(i), iterator});
      parent = iterator;
    }
  }

  // Create an iterator for the result path
  TensorPath resultPath = graph.getResultTensorPath();
  if (resultPath.defined()) {
    TensorVar tensorVar = resultPath.getTensor();
    Format format = tensorVar.getFormat();
    ir::Expr tensorVarExpr = tensorVariables.at(tensorVar);

    storage::Iterator parent = Iterator::makeRoot(tensorVarExpr);
    roots.insert({resultPath, parent});

    for (int i=0; i < (int)tensorVar.getType().getShape().getOrder(); ++i) {
      IndexVar var = tensorVar.getFreeVars()[i];
      Iterator iterator = Iterator::make(var.getName(), tensorVarExpr, i,
                                         format.getModeTypes()[i],
                                         format.getModeOrdering()[i], parent,
                                         tensorVar.getType());
      taco_iassert(resultPath.getStep(i).getStep() == i);
      iterators.insert({resultPath.getStep(i), iterator});
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
bool needsMerge(const std::vector<storage::Iterator>& iterators) {
  int notRandomAccess = 0;
  for (auto& iterator : iterators) {
    if ((!iterator.isRandomAccess()) && (++notRandomAccess > 1)) {
      return true;
    }
  }
  return false;
}

std::vector<storage::Iterator>
getDenseIterators(const std::vector<storage::Iterator>& iterators) {
  vector<storage::Iterator> denseIterators;
  for (auto& iterator : iterators) {
    if (iterator.defined() && iterator.isDense()) {
      denseIterators.push_back(iterator);
    }
  }
  return denseIterators;
}

vector<storage::Iterator>
getSequentialAccessIterators(const vector<storage::Iterator>& iterators) {
  vector<storage::Iterator> sequentialAccessIterators;
  for (auto& iterator : iterators) {
    if (iterator.defined() && iterator.isSequentialAccess()) {
      sequentialAccessIterators.push_back(iterator);
    }
  }
  return sequentialAccessIterators;
}

vector<storage::Iterator>
getRandomAccessIterators(const vector<storage::Iterator>& iterators) {
  vector<storage::Iterator> randomAccessIterators;
  for (auto& iterator : iterators) {
    if (iterator.defined() && iterator.isRandomAccess()) {
      randomAccessIterators.push_back(iterator);
    }
  }
  return randomAccessIterators;
}

vector<ir::Expr> getIdxVars(const vector<storage::Iterator>& iterators) {
  vector<ir::Expr> idxVars;
  for (auto& iterator : iterators) {
    idxVars.push_back(iterator.getIdxVar());
  }
  return idxVars;
}

}}
