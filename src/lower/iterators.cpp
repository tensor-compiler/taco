#include "iterators.h"

#include <iostream>

#include "var.h"
#include "format.h"
#include "internal_tensor.h"
#include "error.h"
#include "ir/ir.h"
#include "util/collections.h"

using namespace std;
using namespace taco::ir;
using taco::internal::Tensor;

namespace taco {
namespace lower {

// class Iterators
Iterators::Iterators(const IterationSchedule& schedule,
                     const map<internal::Tensor,ir::Expr>& tensorVariables) {
  root = storage::Iterator::makeRoot();

  // Create an iterator for each path step
  for (auto& path : schedule.getTensorPaths()) {
    Tensor tensor = path.getTensor();
    ir::Expr tensorVar = tensorVariables.at(tensor);
    Format format = path.getTensor().getFormat();

    storage::Iterator parent = root;
    for (int i=0; i < (int)path.getSize(); ++i) {
      Level levelFormat = format.getLevels()[i];
      string name = path.getVariables()[i].getName();

      storage::Iterator iterator =
          storage::Iterator::make(name, tensorVar, i, levelFormat, parent,
                                  tensor);
      iassert(path.getStep(i).getStep() == i);
      iterators.insert({path.getStep(i), iterator});
      parent = iterator;
    }
  }

  // Create an iterator for the result path
  TensorPath resultPath = schedule.getResultTensorPath();
  if (resultPath.defined()) {
    Tensor tensor = resultPath.getTensor();
    ir::Expr tensorVar = tensorVariables.at(tensor);
    Format format = tensor.getFormat();

    storage::Iterator parent = root;
    for (int i=0; i < (int)format.getLevels().size(); ++i) {
      taco::Var var = tensor.getIndexVars()[i];
      Level levelFormat = format.getLevels()[i];
      string name = var.getName();
      storage::Iterator iterator =
      storage::Iterator::make(name, tensorVar, i, levelFormat, parent, tensor);
      iassert(resultPath.getStep(i).getStep() == i);
      iterators.insert({resultPath.getStep(i), iterator});
      parent = iterator;
    }
  }
}

const storage::Iterator&
Iterators::operator[](const TensorPathStep& step) const {
  iassert(util::contains(iterators, step)) << "No iterator for step: " << step;
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

const storage::Iterator& Iterators::getRoot() const {
  return root;
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
