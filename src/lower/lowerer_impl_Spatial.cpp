#include <taco/lower/mode_format_compressed.h>
#include "taco/lower/lowerer_impl.h"

#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/index_notation/index_notation_visitor.h"
#include "taco/ir/ir.h"
#include "ir/ir_generators.h"
#include "taco/ir/ir_visitor.h"
#include "taco/ir/simplify.h"
#include "taco/lower/iterator.h"
#include "taco/lower/merge_lattice.h"
#include "mode_access.h"
#include "taco/util/collections.h"

using namespace std;
using namespace taco::ir;
using taco::util::combine;

namespace taco {

class LowererImplSpatial::Visitor : public LowererImpl {

}

Stmt
LowererImplSpatial::lower(IndexStmt stmt, string name, 
                   bool assemble, bool compute, bool pack, bool unpack)
{
  this->assemble = assemble;
  this->compute = compute;
  definedIndexVarsOrdered = {};
  definedIndexVars = {};

  // Create result and parameter variables
  vector<TensorVar> results = getResults(stmt);
  vector<TensorVar> arguments = getArguments(stmt);
  vector<TensorVar> temporaries = getTemporaries(stmt);

  // Convert tensor results and arguments IR variables
  map<TensorVar, Expr> resultVars;
  vector<Expr> resultsIR = createVars(results, &resultVars, unpack);
  tensorVars.insert(resultVars.begin(), resultVars.end());
  vector<Expr> argumentsIR = createVars(arguments, &tensorVars, pack);

  // Create variables for temporaries
  // TODO Remove this
  for (auto& temp : temporaries) {
    ir::Expr irVar = ir::Var::make(temp.getName(), temp.getType().getDataType(),
                                   true, true);
    tensorVars.insert({temp, irVar});
  }

  // Create variables for keeping track of result values array capacity
  createCapacityVars(resultVars, &capacityVars);

  // Create iterators
  iterators = Iterators(stmt, tensorVars);

  provGraph = ProvenanceGraph(stmt);

  for (const IndexVar indexVar : provGraph.getAllIndexVars()) {
    if (iterators.modeIterators().count(indexVar)) {
      indexVarToExprMap.insert({indexVar, iterators.modeIterators()[indexVar].getIteratorVar()});
    }
    else {
      indexVarToExprMap.insert({indexVar, Var::make(indexVar.getName(), Int())});
    }
  }

  vector<Access> inputAccesses, resultAccesses;
  set<Access> reducedAccesses;
  inputAccesses = getArgumentAccesses(stmt);
  std::tie(resultAccesses, reducedAccesses) = getResultAccesses(stmt);

  // Create variables that represent the reduced values of duplicated tensor 
  // components
  createReducedValueVars(inputAccesses, &reducedValueVars);

  map<TensorVar, Expr> scalars;

  // Define and initialize dimension variables
  vector<IndexVar> indexVars = getIndexVars(stmt);
  for (auto& indexVar : indexVars) {
    Expr dimension;
    match(stmt,
      function<void(const AssignmentNode*, Matcher*)>([&](
          const AssignmentNode* n, Matcher* m) {
        m->match(n->rhs);
        if (!dimension.defined()) {
          auto ivars = n->lhs.getIndexVars();
          int loc = (int)distance(ivars.begin(),
                                  find(ivars.begin(),ivars.end(), indexVar));
          dimension = GetProperty::make(tensorVars.at(n->lhs.getTensorVar()),
                                        TensorProperty::Dimension, loc);
        }
      }),
      function<void(const AccessNode*)>([&](const AccessNode* n) {
        auto indexVars = n->indexVars;
        if (util::contains(indexVars, indexVar)) {
          int loc = (int)distance(indexVars.begin(),
                                  find(indexVars.begin(),indexVars.end(),
                                       indexVar));
          dimension = GetProperty::make(tensorVars.at(n->tensorVar),
                                        TensorProperty::Dimension, loc);
        }
      })
    );
    dimensions.insert({indexVar, dimension});
    underivedBounds.insert({indexVar, {ir::Literal::make(0), dimension}});
  }

  // Define and initialize scalar results and arguments
  if (generateComputeCode()) {
    for (auto& result : results) {
      if (isScalar(result.getType())) {
        taco_iassert(!util::contains(scalars, result));
        taco_iassert(util::contains(tensorVars, result));
        scalars.insert({result, tensorVars.at(result)});
        header.push_back(defineScalarVariable(result, true));
      }
    }
    for (auto& argument : arguments) {
      if (isScalar(argument.getType())) {
        taco_iassert(!util::contains(scalars, argument));
        taco_iassert(util::contains(tensorVars, argument));
        scalars.insert({argument, tensorVars.at(argument)});
        header.push_back(defineScalarVariable(argument, false));
      }
    }
  }

  // Allocate memory for scalar results
  if (generateAssembleCode()) {
    for (auto& result : results) {
      if (result.getOrder() == 0) {
        Expr resultIR = resultVars.at(result);
        Expr vals = GetProperty::make(resultIR, TensorProperty::Values);
        header.push_back(Allocate::make(vals, 1));
      }
    }
  }

  // Allocate and initialize append and insert mode indices
  Stmt initializeResults = initResultArrays(resultAccesses, inputAccesses, 
                                            reducedAccesses);

  // Lower the index statement to compute and/or assemble
  Stmt body = lower(stmt);

  // Post-process result modes and allocate memory for values if necessary
  Stmt finalizeResults = finalizeResultArrays(resultAccesses);

  // Store scalar stack variables back to results
  if (generateComputeCode()) {
    for (auto& result : results) {
      if (isScalar(result.getType())) {
        taco_iassert(util::contains(scalars, result));
        taco_iassert(util::contains(tensorVars, result));
        Expr resultIR = scalars.at(result);
        Expr varValueIR = tensorVars.at(result);
        Expr valuesArrIR = GetProperty::make(resultIR, TensorProperty::Values);
        footer.push_back(Store::make(valuesArrIR, 0, varValueIR, markAssignsAtomicDepth > 0, atomicParallelUnit));
      }
    }
  }

  // Create function
  return Function::make(name, resultsIR, argumentsIR,
                        Block::blanks(Block::make(header),
                                      initializeResults,
                                      body,
                                      finalizeResults,
                                      Block::make(footer)));
}

