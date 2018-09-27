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

namespace taco {
namespace old {

// class Iterators
Iterators::Iterators() {
}

Iterators::Iterators(const IterationGraph& graph,
                     const map<TensorVar,ir::Expr>& tensorVariables) {
  // Create an iterator for each path step
  for (TensorPath path : util::combine(graph.getTensorPaths(),
                                       {graph.getResultTensorPath()})) {
    TensorVar tensorVar = path.getAccess().getTensorVar();
    Shape shape = tensorVar.getType().getShape();
    Format format = tensorVar.getFormat();
    ir::Expr tensorVarExpr = tensorVariables.at(tensorVar);

    Iterator parent(tensorVarExpr);
    roots.insert({path, parent});

    ModeFormat parentModeType;

    taco_iassert(path.getSize() == (size_t)format.getOrder());
    int level = 1;
    for (ModeFormatPack modeTypePack : format.getModeFormatPacks()) {
      vector<Expr> arrays;
      taco_iassert(modeTypePack.getModeFormats().size() > 0);

      ModePack modePack(modeTypePack.getModeFormats().size(),
                        modeTypePack.getModeFormats()[0], tensorVarExpr, level);

      int pos = 0;
      for (auto& modeType : modeTypePack.getModeFormats()) {
        Dimension dim = shape.getDimension(format.getModeOrdering()[level-1]);
        Mode mode(tensorVarExpr, dim, level, modeType, modePack, pos,
                  parentModeType);

        taco_iassert(path.getStep(level-1).getStep() == level-1);
        std::string indexVarName = path.getVariables()[level-1].getName();
        Iterator iterator(path, indexVarName, tensorVarExpr, mode, parent);
        iterators.insert({path.getStep(level-1), iterator});
        parent = iterator;

        parentModeType = modeType;
        pos++;
        level++;
      }
    }
  }
}

const Iterator& Iterators::operator[](const TensorPathStep& step) const {
  taco_iassert(util::contains(iterators, step)) <<
      "No iterator for step: " << step;
  return iterators.at(step);
}

vector<Iterator>
Iterators::operator[](const vector<TensorPathStep>& steps) const {
  vector<Iterator> iterators;
  for (auto& step : steps) {
    iterators.push_back((*this)[step]);
  }
  return iterators;
}

const Iterator& Iterators::getRoot(const TensorPath& path) const {
  taco_iassert(util::contains(roots, path)) <<
      path << " does not have a root iterator";
  return roots.at(path);
}


// functions
std::vector<Iterator>
getFullIterators(const std::vector<Iterator>& iterators) {
  vector<Iterator> fullIterators;
  for (auto& iterator : iterators) {
    if (iterator.defined() && iterator.isFull()) {
      fullIterators.push_back(iterator);
    }
  }
  return fullIterators;
}

vector<ir::Expr> getIdxVars(const vector<Iterator>& iterators) {
  vector<ir::Expr> idxVars;
  for (auto& iterator : iterators) {
    idxVars.push_back(iterator.getCoordVar());
  }
  return idxVars;
}

}}
