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

class LowererImpl::Visitor : public IndexNotationVisitorStrict {
public:
  Visitor(LowererImpl* impl) : impl(impl) {}
  Stmt lower(IndexStmt stmt) {
    this->stmt = Stmt();
    impl->accessibleIterators.scope();
    IndexStmtVisitorStrict::visit(stmt);
    impl->accessibleIterators.unscope();
    return this->stmt;
  }
  Expr lower(IndexExpr expr) {
    this->expr = Expr();
    IndexExprVisitorStrict::visit(expr);
    return this->expr;
  }
private:
  LowererImpl* impl;
  Expr expr;
  Stmt stmt;
  using IndexNotationVisitorStrict::visit;
  void visit(const AssignmentNode* node)    { stmt = impl->lowerAssignment(node); }
  void visit(const YieldNode* node)         { stmt = impl->lowerYield(node); }
  void visit(const ForallNode* node)        { stmt = impl->lowerForall(node); }
  void visit(const WhereNode* node)         { stmt = impl->lowerWhere(node); }
  void visit(const MultiNode* node)         { stmt = impl->lowerMulti(node); }
  void visit(const SuchThatNode* node)      { stmt = impl->lowerSuchThat(node); }
  void visit(const SequenceNode* node)      { stmt = impl->lowerSequence(node); }
  void visit(const AccessNode* node)        { expr = impl->lowerAccess(node); }
  void visit(const LiteralNode* node)       { expr = impl->lowerLiteral(node); }
  void visit(const NegNode* node)           { expr = impl->lowerNeg(node); }
  void visit(const AddNode* node)           { expr = impl->lowerAdd(node); }
  void visit(const SubNode* node)           { expr = impl->lowerSub(node); }
  void visit(const MulNode* node)           { expr = impl->lowerMul(node); }
  void visit(const DivNode* node)           { expr = impl->lowerDiv(node); }
  void visit(const SqrtNode* node)          { expr = impl->lowerSqrt(node); }
  void visit(const CastNode* node)          { expr = impl->lowerCast(node); }
  void visit(const CallIntrinsicNode* node) { expr = impl->lowerCallIntrinsic(node); }
  void visit(const ReductionNode* node)  {
    taco_ierror << "Reduction nodes not supported in concrete index notation";
  }
};

LowererImpl::LowererImpl() : visitor(new Visitor(this)) {
}


static void createCapacityVars(const map<TensorVar, Expr>& tensorVars,
                               map<Expr, Expr>* capacityVars) {
  for (auto& tensorVar : tensorVars) {
    Expr tensor = tensorVar.second;
    Expr capacityVar = Var::make(util::toString(tensor) + "_capacity", Int());
    capacityVars->insert({tensor, capacityVar});
  }
}

static void createReducedValueVars(const vector<Access>& inputAccesses,
                                   map<Access, Expr>* reducedValueVars) {
  for (const auto& access : inputAccesses) {
    const TensorVar inputTensor = access.getTensorVar();
    const std::string name = inputTensor.getName() + "_val";
    const Datatype type = inputTensor.getType().getDataType();
    reducedValueVars->insert({access, Var::make(name, type)});
  }
}

/// Returns true iff `stmt` modifies an array
static bool hasStores(Stmt stmt) {
  struct FindStores : IRVisitor {
    bool hasStore;

    using IRVisitor::visit;

    void visit(const Store* stmt) {
      hasStore = true;
    }

    bool hasStores(Stmt stmt) {
      hasStore = false;
      stmt.accept(this);
      return hasStore;
    }
  };
  return stmt.defined() && FindStores().hasStores(stmt);
}

Stmt
LowererImpl::lower(IndexStmt stmt, string name, bool assemble, bool compute)
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
  vector<Expr> resultsIR = createVars(results, &resultVars);
  tensorVars.insert(resultVars.begin(), resultVars.end());
  vector<Expr> argumentsIR = createVars(arguments, &tensorVars);

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

  relGraph = IndexVarRelGraph(stmt);

  for (const IndexVar indexVar : relGraph.getAllIndexVars()) {
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


Stmt LowererImpl::lowerAssignment(Assignment assignment)
{
  TensorVar result = assignment.getLhs().getTensorVar();

  if (generateComputeCode()) {
    Expr var = getTensorVar(result);
    Expr rhs = lower(assignment.getRhs());

    // Assignment to scalar variables.
    if (isScalar(result.getType())) {
      if (!assignment.getOperator().defined()) {
        return Assign::make(var, rhs, markAssignsAtomicDepth > 0 && !util::contains(whereTemps, result), atomicParallelUnit);
        // TODO: we don't need to mark all assigns/stores just when scattering/reducing
      }
      else {
        taco_iassert(isa<taco::Add>(assignment.getOperator()));
        return compoundAssign(var, rhs, markAssignsAtomicDepth > 0 && !util::contains(whereTemps, result), atomicParallelUnit);
      }
    }
    // Assignments to tensor variables (non-scalar).
    else {
      Expr values = getValuesArray(result);
      Expr loc = generateValueLocExpr(assignment.getLhs());

      Stmt computeStmt;
      if (!assignment.getOperator().defined()) {
        computeStmt = Store::make(values, loc, rhs, markAssignsAtomicDepth > 0, atomicParallelUnit);
      }
      else {
        computeStmt = compoundStore(values, loc, rhs, markAssignsAtomicDepth > 0, atomicParallelUnit);
      }
      taco_iassert(computeStmt.defined());
      return computeStmt;
    }
  }
  // We're only assembling so defer allocating value memory to the end when
  // we'll know exactly how much we need.
  else if (generateAssembleCode()) {
    // TODO
    return Stmt();
  }
  // We're neither assembling or computing so we emit nothing.
  else {
    return Stmt();
  }
  taco_unreachable;
  return Stmt();
}


Stmt LowererImpl::lowerYield(Yield yield) {
  std::vector<Expr> coords;
  for (auto& indexVar : yield.getIndexVars()) {
    coords.push_back(getCoordinateVar(indexVar));
  }
  Expr val = lower(yield.getExpr());
  return ir::Yield::make(coords, val);
}


static pair<vector<Iterator>, vector<Iterator>>
splitAppenderAndInserters(const vector<Iterator>& results) {
  vector<Iterator> appenders;
  vector<Iterator> inserters;

  // TODO: Choose insert when the current forall is nested inside a reduction
  for (auto& result : results) {
    taco_iassert(result.hasAppend() || result.hasInsert())
        << "Results must support append or insert";

    if (result.hasAppend()) {
      appenders.push_back(result);
    }
    else {
      taco_iassert(result.hasInsert());
      inserters.push_back(result);
    }
  }

  return {appenders, inserters};
}


Stmt LowererImpl::lowerForall(Forall forall)
{
  bool hasExactBound = relGraph.hasExactBound(forall.getIndexVar()); // TODO: check for evenly divisible too
  if (hasExactBound) {
    emitUnderivedGuards = false;
  }
  if (!ignoreVectorize && emitUnderivedGuards && (forall.getParallelUnit() == PARALLEL_UNIT::CPU_VECTOR || forall.getUnrollFactor() > 0)) {
    // want to emit guards outside of loop to prevent unstructured loop exits

    // construct guard
    // underived or pos variables that have a descendant that has not been defined yet
    vector<IndexVar> varsWithGuard;
    for (auto var : relGraph.getAllIndexVars()) {
      if (relGraph.isRecoverable(var, definedIndexVars)) {
        continue; // already recovered
      }
      if (relGraph.isUnderived(var) && !relGraph.hasPosDescendant(var)) { // if there is pos descendant then will be guarded already
        varsWithGuard.push_back(var);
      }
      else if (relGraph.isPosVariable(var)) {
        // if parent is coord then this is variable that will be guarded when indexing into coord array
        if(relGraph.getParents(var).size() == 1 && relGraph.isCoordVariable(relGraph.getParents(var)[0])) {
          varsWithGuard.push_back(var);
        }
      }
    }

    // determine min and max values for vars given already defined variables.
    // we do a recovery where we fill in undefined variables with either 0's or the max of their iteration
    std::map<IndexVar, Expr> minVarValues;
    std::map<IndexVar, Expr> maxVarValues;
    set<IndexVar> definedForGuard = definedIndexVars;
    vector<Stmt> guardRecoverySteps;

    // TODO:
    for (auto var : varsWithGuard) {
      std::vector<IndexVar> currentDefinedVarOrder = definedIndexVarsOrdered; // TODO: get defined vars at time of this recovery

      std::map<IndexVar, Expr> minChildValues = indexVarToExprMap;
      std::map<IndexVar, Expr> maxChildValues = indexVarToExprMap;

      for (auto child : relGraph.getFullyDerivedDescendants(var)) {
        if (!definedIndexVars.count(child)) {
          std::vector<ir::Expr> childBounds = relGraph.deriveIterBounds(child, currentDefinedVarOrder, underivedBounds, indexVarToExprMap, iterators);

          minChildValues[child] = childBounds[0];
          maxChildValues[child] = childBounds[1];

          // recover new parents
          for (const IndexVar& varToRecover : relGraph.newlyRecoverableParents(child, definedForGuard)) {

            std::map<IndexVar, Expr> *varValues = &minChildValues;
            if (relGraph.childRelMap.count(varToRecover) && relGraph.childRelMap.at(varToRecover).getRelType() == PRECOMPUTE) {
              varValues = &maxChildValues;
            }

            Expr recoveredValue = relGraph.recoverVariable(varToRecover, definedIndexVarsOrdered, underivedBounds,
                                                           *varValues, iterators); // TODO: should be min and max
            taco_iassert(indexVarToExprMap.count(varToRecover));

            if (relGraph.childRelMap.count(varToRecover) && relGraph.childRelMap.at(varToRecover).getRelType() == PRECOMPUTE) {
              guardRecoverySteps.push_back(VarDecl::make(indexVarToExprMap[varToRecover], ir::Sub::make(recoveredValue, 1))); // TODO
            }
            else {
              guardRecoverySteps.push_back(VarDecl::make(indexVarToExprMap[varToRecover], recoveredValue)); // TODO
            }
            definedForGuard.insert(varToRecover);
            if (varToRecover == var) {
              break;
            }
          }
          definedForGuard.insert(child);
        }
      }

      minVarValues[var] = relGraph.recoverVariable(var, currentDefinedVarOrder, underivedBounds, minChildValues, iterators);
      maxVarValues[var] = relGraph.recoverVariable(var, currentDefinedVarOrder, underivedBounds, maxChildValues, iterators);
    }

    // Build guards
    Expr guardCondition;
    for (auto var : varsWithGuard) {
      std::vector<ir::Expr> iterBounds = relGraph.deriveIterBounds(var, definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);

      Expr minGuard = Lt::make(minVarValues[var], iterBounds[0]);
      Expr maxGuard = Gte::make(maxVarValues[var], iterBounds[1]);
      Expr guardConditionCurrent = Or::make(minGuard, maxGuard);

      if (isa<ir::Literal>(ir::simplify(iterBounds[0])) && ir::simplify(iterBounds[0]).as<ir::Literal>()->equalsScalar(0)) {
        guardConditionCurrent = maxGuard;
      }

      if (guardCondition.defined()) {
        guardCondition = Or::make(guardConditionCurrent, guardCondition);
      }
      else {
        guardCondition = guardConditionCurrent;
      }
    }

    Stmt unvectorizedLoop;
    // build loop with guards (not vectorized)
    if (!varsWithGuard.empty()) {
      ignoreVectorize = true;
      unvectorizedLoop = lowerForall(forall);
      ignoreVectorize = false;
    }

    // build loop without guards
    emitUnderivedGuards = false;
    Stmt vectorizedLoop = lowerForall(forall);
    emitUnderivedGuards = true;

    // return guarded loops
    return Block::make(Block::make(guardRecoverySteps), IfThenElse::make(guardCondition, unvectorizedLoop, vectorizedLoop));
  }

  if (forall.getParallelUnit() != PARALLEL_UNIT::NOT_PARALLEL) {
    inParallelLoopDepth++;
  }

  // Recover any available parents that were not recoverable previously
  vector<Stmt> recoverySteps;
  for (const IndexVar& varToRecover : relGraph.newlyRecoverableParents(forall.getIndexVar(), definedIndexVars)) {
    // place pos guard
    if (emitUnderivedGuards && relGraph.isCoordVariable(varToRecover)
    && relGraph.getChildren(varToRecover).size() == 1 && relGraph.isPosVariable(relGraph.getChildren(varToRecover)[0])) {
      IndexVar posVar = relGraph.getChildren(varToRecover)[0];
      std::vector<ir::Expr> iterBounds = relGraph.deriveIterBounds(posVar, definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);

      Expr minGuard = Lt::make(indexVarToExprMap[posVar], iterBounds[0]);
      Expr maxGuard = Gte::make(indexVarToExprMap[posVar], iterBounds[1]);
      Expr guardCondition = Or::make(minGuard, maxGuard);
      if (isa<ir::Literal>(ir::simplify(iterBounds[0])) && ir::simplify(iterBounds[0]).as<ir::Literal>()->equalsScalar(0)) {
        guardCondition = maxGuard;
      }
      ir::Stmt guard = ir::IfThenElse::make(guardCondition, ir::Break::make());
      recoverySteps.push_back(guard);
    }

    Expr recoveredValue = relGraph.recoverVariable(varToRecover, definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
    taco_iassert(indexVarToExprMap.count(varToRecover));
    recoverySteps.push_back(VarDecl::make(indexVarToExprMap[varToRecover], recoveredValue));
    // place underived guard
    if (emitUnderivedGuards && underivedBounds.count(varToRecover) && !relGraph.hasPosDescendant(varToRecover)) {
      Stmt guard = IfThenElse::make(Gte::make(indexVarToExprMap[varToRecover], underivedBounds[varToRecover][1]),
                                    Break::make());
      recoverySteps.push_back(guard);
    }
  }
  Stmt recoveryStmt = Block::make(recoverySteps);

  taco_iassert(!definedIndexVars.count(forall.getIndexVar()));
  definedIndexVars.insert(forall.getIndexVar());
  definedIndexVarsOrdered.push_back(forall.getIndexVar());

  if (forall.getParallelUnit() != PARALLEL_UNIT::NOT_PARALLEL) {
    taco_iassert(!parallelUnitSizes.count(forall.getParallelUnit()));
    taco_iassert(!parallelUnitIndexVars.count(forall.getParallelUnit()));
    parallelUnitIndexVars[forall.getParallelUnit()] = forall.getIndexVar();
    vector<Expr> bounds = relGraph.deriveIterBounds(forall.getIndexVar(), definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
    parallelUnitSizes[forall.getParallelUnit()] = ir::Sub::make(bounds[1], bounds[0]);
  }

  MergeLattice lattice = MergeLattice::make(forall, iterators, relGraph, definedIndexVars, whereTempsToResult);
  vector<Access> resultAccesses;
  set<Access> reducedAccesses;
  std::tie(resultAccesses, reducedAccesses) = getResultAccesses(forall);

  // Pre-allocate/initialize memory of value arrays that are full below this
  // loops index variable
  Stmt preInitValues = initResultArrays(forall.getIndexVar(), resultAccesses,
                                        getArgumentAccesses(forall), 
                                        reducedAccesses);

  Stmt loops;
  // Emit a loop that iterates over over a single iterator (optimization)
  if (lattice.iterators().size() == 1 && lattice.iterators()[0].isUnique()) {
    taco_iassert(lattice.points().size() == 1);

    MergePoint point = lattice.points()[0];
    Iterator iterator = lattice.iterators()[0];

    vector<Iterator> locators = point.locators();
    vector<Iterator> appenders;
    vector<Iterator> inserters;
    tie(appenders, inserters) = splitAppenderAndInserters(point.results());

    std::vector<IndexVar> underivedAncestors = relGraph.getUnderivedAncestors(iterator.getIndexVar());
    IndexVar posDescendant;
    bool hasPosDescendant = false;
    if (!underivedAncestors.empty()) {
      hasPosDescendant = relGraph.getPosIteratorFullyDerivedDescendant(underivedAncestors[0], &posDescendant);
    }

    bool isWhereProducer = false;
    vector<Iterator> results = point.results();
    for (Iterator result : results) {
      for (auto it = tensorVars.begin(); it != tensorVars.end(); it++) {
        if (it->second == result.getTensor()) {
          if (whereTempsToResult.count(it->first)) {
            isWhereProducer = true;
            break;
          }
        }
      }
    }
    if (!isWhereProducer && hasPosDescendant && underivedAncestors.size() > 1 && relGraph.isPosVariable(iterator.getIndexVar()) && posDescendant == forall.getIndexVar()) {
      loops = lowerForallFusedPosition(forall, iterator, locators,
                                         inserters, appenders, reducedAccesses, recoveryStmt);
    }
    // Emit dimension coordinate iteration loop
    else if (iterator.isDimensionIterator()) {
      loops = lowerForallDimension(forall, point.locators(),
                                   inserters, appenders, reducedAccesses, recoveryStmt);
    }
    // Emit position iteration loop
    else if (iterator.hasPosIter()) {
      loops = lowerForallPosition(forall, iterator, locators,
                                    inserters, appenders, reducedAccesses, recoveryStmt);
    }
    // Emit coordinate iteration loop
    else {
      taco_iassert(iterator.hasCoordIter());
//      taco_not_supported_yet
      loops = Stmt();
    }
  }
  // Emit general loops to merge multiple iterators
  else {
    std::vector<IndexVar> underivedAncestors = relGraph.getUnderivedAncestors(forall.getIndexVar());
    taco_iassert(underivedAncestors.size() == 1); // TODO:
    loops = lowerMergeLattice(lattice, underivedAncestors[0],
                              forall.getStmt(), reducedAccesses);
  }
//  taco_iassert(loops.defined());

  if (!generateComputeCode() && !hasStores(loops)) {
    // If assembly loop does not modify output arrays, then it can be safely 
    // omitted.
    loops = Stmt();
  }
  definedIndexVars.erase(forall.getIndexVar());
  definedIndexVarsOrdered.pop_back();
  if (forall.getParallelUnit() != PARALLEL_UNIT::NOT_PARALLEL) {
    inParallelLoopDepth--;
    taco_iassert(parallelUnitSizes.count(forall.getParallelUnit()));
    taco_iassert(parallelUnitIndexVars.count(forall.getParallelUnit()));
    parallelUnitIndexVars.erase(forall.getParallelUnit());
    parallelUnitSizes.erase(forall.getParallelUnit());
  }
  return Block::blanks(preInitValues,
                       loops);
}


Stmt LowererImpl::lowerForallDimension(Forall forall,
                                       vector<Iterator> locators,
                                       vector<Iterator> inserters,
                                       vector<Iterator> appenders,
                                       set<Access> reducedAccesses,
                                       ir::Stmt recoveryStmt)
{
  Expr coordinate = getCoordinateVar(forall.getIndexVar());

  if (forall.getParallelUnit() != PARALLEL_UNIT::NOT_PARALLEL && forall.getOutputRaceStrategy() == OUTPUT_RACE_STRATEGY::ATOMICS) {
    markAssignsAtomicDepth++;
    atomicParallelUnit = forall.getParallelUnit();
  }

  Stmt body = lowerForallBody(coordinate, forall.getStmt(),
                              locators, inserters, appenders, reducedAccesses);

  if (forall.getParallelUnit() != PARALLEL_UNIT::NOT_PARALLEL && forall.getOutputRaceStrategy() == OUTPUT_RACE_STRATEGY::ATOMICS) {
    markAssignsAtomicDepth--;
  }

  body = Block::make({recoveryStmt, body});

  Stmt posAppend = generateAppendPositions(appenders);

  // Emit loop with preamble and postamble
  std::vector<ir::Expr> bounds = relGraph.deriveIterBounds(forall.getIndexVar(), definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);

  LoopKind kind = LoopKind::Serial;
  if (forall.getParallelUnit() == PARALLEL_UNIT::CPU_VECTOR && !ignoreVectorize) {
    kind = LoopKind::Vectorized;
  }
  else if (forall.getParallelUnit() != PARALLEL_UNIT::NOT_PARALLEL
            && forall.getOutputRaceStrategy() != OUTPUT_RACE_STRATEGY::PARALLEL_REDUCTION && !ignoreVectorize) {
    kind = LoopKind::Runtime;
  }

  return Block::blanks(For::make(coordinate, bounds[0], bounds[1], 1, body,
                                 kind,
                                 ignoreVectorize ? PARALLEL_UNIT::NOT_PARALLEL : forall.getParallelUnit(), ignoreVectorize ? 0 : forall.getUnrollFactor()),
                       posAppend);
}


Stmt LowererImpl::lowerForallCoordinate(Forall forall, Iterator iterator,
                                        vector<Iterator> locators,
                                        vector<Iterator> inserters,
                                        vector<Iterator> appenders,
                                        set<Access> reducedAccesses,
                                        ir::Stmt recoveryStmt) {
  taco_not_supported_yet;
  return Stmt();
}

Stmt LowererImpl::lowerForallPosition(Forall forall, Iterator iterator,
                                      vector<Iterator> locators,
                                      vector<Iterator> inserters,
                                      vector<Iterator> appenders,
                                      set<Access> reducedAccesses,
                                      ir::Stmt recoveryStmt)
{
  Expr coordinate = getCoordinateVar(forall.getIndexVar());
  Stmt declareCoordinate = Stmt();
  if (relGraph.isCoordVariable(forall.getIndexVar())) {
    Expr coordinateArray = iterator.posAccess(iterator.getPosVar(),
                                              coordinates(iterator)).getResults()[0];
    declareCoordinate = VarDecl::make(coordinate, coordinateArray);
  }
  if (forall.getParallelUnit() != PARALLEL_UNIT::NOT_PARALLEL && forall.getOutputRaceStrategy() == OUTPUT_RACE_STRATEGY::ATOMICS) {
    markAssignsAtomicDepth++;
  }

  Stmt body = lowerForallBody(coordinate, forall.getStmt(),
                              locators, inserters, appenders, reducedAccesses);

  if (forall.getParallelUnit() != PARALLEL_UNIT::NOT_PARALLEL && forall.getOutputRaceStrategy() == OUTPUT_RACE_STRATEGY::ATOMICS) {
    markAssignsAtomicDepth--;
  }

  body = Block::make(recoveryStmt, body);

  // Code to append positions
  Stmt posAppend = generateAppendPositions(appenders);

  // Code to compute iteration bounds
  Stmt boundsCompute;
  Expr startBound, endBound;
  Expr parentPos = iterator.getParent().getPosVar();
  if (!relGraph.isUnderived(iterator.getIndexVar())) {
    vector<Expr> bounds = relGraph.deriveIterBounds(iterator.getIndexVar(), definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
    startBound = bounds[0];
    endBound = bounds[1];
  }
  else if (iterator.getParent().isRoot() || iterator.getParent().isUnique()) {
    // E.g. a compressed mode without duplicates
    ModeFunction bounds = iterator.posBounds(parentPos);
    boundsCompute = bounds.compute();
    startBound = bounds[0];
    endBound = bounds[1];
  } else {
    taco_iassert(iterator.isOrdered() && iterator.getParent().isOrdered());
    taco_iassert(iterator.isCompact() && iterator.getParent().isCompact());

    // E.g. a compressed mode with duplicates. Apply iterator chaining
    Expr parentSegend = iterator.getParent().getSegendVar();
    ModeFunction startBounds = iterator.posBounds(parentPos);
    ModeFunction endBounds = iterator.posBounds(ir::Sub::make(parentSegend, 1));
    boundsCompute = Block::make(startBounds.compute(), endBounds.compute());
    startBound = startBounds[0];
    endBound = endBounds[1];
  }

  LoopKind kind = LoopKind::Serial;
  if (forall.getParallelUnit() == PARALLEL_UNIT::CPU_VECTOR && !ignoreVectorize) {
    kind = LoopKind::Vectorized;
  }
  else if (forall.getParallelUnit() != PARALLEL_UNIT::NOT_PARALLEL
           && forall.getOutputRaceStrategy() != OUTPUT_RACE_STRATEGY::PARALLEL_REDUCTION && !ignoreVectorize) {
    kind = LoopKind::Runtime;
  }
  // Loop with preamble and postamble
  return Block::blanks(boundsCompute,
                       For::make(iterator.getPosVar(), startBound, endBound, 1,
                                 Block::make(declareCoordinate, body),
                                 kind,
                                 ignoreVectorize ? PARALLEL_UNIT::NOT_PARALLEL : forall.getParallelUnit(), ignoreVectorize ? 0 : forall.getUnrollFactor()),
                       posAppend);

}

Stmt LowererImpl::lowerForallFusedPosition(Forall forall, Iterator iterator,
                                      vector<Iterator> locators,
                                      vector<Iterator> inserters,
                                      vector<Iterator> appenders,
                                      set<Access> reducedAccesses,
                                      ir::Stmt recoveryStmt)
{
  Expr coordinate = getCoordinateVar(forall.getIndexVar());
  Stmt declareCoordinate = Stmt();
  if (relGraph.isCoordVariable(forall.getIndexVar())) {
    Expr coordinateArray = iterator.posAccess(iterator.getPosVar(),
                                              coordinates(iterator)).getResults()[0];
    declareCoordinate = VarDecl::make(coordinate, coordinateArray);
  }

  // declare upper-level underived ancestors that will be tracked with while loops
  Expr writeResultCond;
  vector<Stmt> loopsToTrackUnderived;
  vector<Stmt> searchForUnderivedStart;
  std::map<IndexVar, vector<Expr>> coordinateBounds = relGraph.deriveCoordBounds(definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
  vector<IndexVar> underivedAncestors = relGraph.getUnderivedAncestors(forall.getIndexVar());
  if (underivedAncestors.size() > 1) {
    // each underived ancestor is initialized to min coordinate bound
    IndexVar posIteratorVar;
    taco_iassert(relGraph.getPosIteratorAncestor(iterator.getIndexVar(), &posIteratorVar));
    // get pos variable then search for leveliterators to find the corresponding iterator

    Iterator posIterator;
    auto iteratorMap = iterators.levelIterators();
    int modePos = -1; // select lowest level possible
    for (auto it = iteratorMap.begin(); it != iteratorMap.end(); it++) {
      if (it->second.getIndexVar() == posIteratorVar && (int) it->first.getModePos() > modePos) {
        posIterator = it->second;
        modePos = (int) it->first.getModePos();
      }
    }
    taco_iassert(posIterator.hasPosIter());

    if (inParallelLoopDepth == 0) {
      for (int i = 0; i < (int) underivedAncestors.size() - 1; i ++) {
        // TODO: only if level is sparse emit underived_pos
        header.push_back(VarDecl::make(this->iterators.modeIterator(underivedAncestors[i]).getPosVar(), 0)); // TODO: set to start position bound
        header.push_back(VarDecl::make(getCoordinateVar(underivedAncestors[i]), coordinateBounds[underivedAncestors[i]][0]));
      }
    } else {
      ir::Expr last_block_start_temporary;
      for (int i = (int) underivedAncestors.size() - 2; i >= 0; i--) {
        Iterator posIteratorLevel = posIterator;
        for (int j = (int) underivedAncestors.size() - 2; j > i; j--) { // take parent of iterator enough times to get correct level
          posIteratorLevel = posIteratorLevel.getParent();
        }

        // want to get size of pos array not of crd_array
        ir::Expr parentSize = 1; // to find size of segment walk down sizes of iterator chain
        Iterator rootIterator = posIterator;
        while (!rootIterator.isRoot()) {
          rootIterator = rootIterator.getParent();
        }
        while (rootIterator.getChild() != posIteratorLevel) {
          rootIterator = rootIterator.getChild();
          if (rootIterator.hasAppend()) {
            parentSize = rootIterator.getSize(parentSize);
          } else if (rootIterator.hasInsert()) {
            parentSize = ir::Mul::make(parentSize, rootIterator.getWidth());
          }
        }

        // emit bounds search on cpu just bounds, on gpu search in blocks
        if (parallelUnitIndexVars.count(PARALLEL_UNIT::GPU_BLOCK)) {
          Expr values_per_block;
          {
            // we do a recovery where we fill in undefined variables with 0's to get start target (just like for vector guards)
            // TODO: this method should really be moved to separate function and reused
            std::map<IndexVar, Expr> zeroedChildValues = indexVarToExprMap;
            zeroedChildValues[parallelUnitIndexVars[PARALLEL_UNIT::GPU_BLOCK]] = 1;
            set<IndexVar> zeroDefinedIndexVars = {parallelUnitIndexVars[PARALLEL_UNIT::GPU_BLOCK]};
            for (IndexVar child : relGraph.getFullyDerivedDescendants(posIterator.getIndexVar())) {
              if (child != parallelUnitIndexVars[PARALLEL_UNIT::GPU_BLOCK]) {
                zeroedChildValues[child] = 0;

                // recover new parents
                for (const IndexVar &varToRecover : relGraph.newlyRecoverableParents(child, zeroDefinedIndexVars)) {
                  Expr recoveredValue = relGraph.recoverVariable(varToRecover, definedIndexVarsOrdered, underivedBounds,
                                                                 zeroedChildValues, iterators);
                  taco_iassert(indexVarToExprMap.count(varToRecover));
                  zeroedChildValues[varToRecover] = recoveredValue;
                  zeroDefinedIndexVars.insert(varToRecover);
                  if (varToRecover == posIterator.getIndexVar()) {
                    break;
                  }
                }
                zeroDefinedIndexVars.insert(child);
              }
            }
            values_per_block = zeroedChildValues[posIterator.getIndexVar()];
          }

          IndexVar underived = underivedAncestors[i];
          ir::Expr blockStarts_temporary = ir::Var::make(underived.getName() + "_blockStarts",
                                                         getCoordinateVar(underived).type(), true, false);
          header.push_back(ir::VarDecl::make(blockStarts_temporary, 0));
          header.push_back(
                  Allocate::make(blockStarts_temporary, ir::Add::make(parallelUnitSizes[PARALLEL_UNIT::GPU_BLOCK], 1)));
          footer.push_back(Free::make(blockStarts_temporary));


          Expr blockSize;
          if (parallelUnitSizes.count(PARALLEL_UNIT::GPU_THREAD)) {
            blockSize = parallelUnitSizes[PARALLEL_UNIT::GPU_THREAD];
            if (parallelUnitSizes.count(PARALLEL_UNIT::GPU_WARP)) {
              blockSize = ir::Mul::make(blockSize, parallelUnitSizes[PARALLEL_UNIT::GPU_WARP]);
            }
          } else {
            std::vector<IndexVar> definedIndexVarsMatched = definedIndexVarsOrdered;
            // find sub forall that tells us block size
            match(forall.getStmt(),
                  function<void(const ForallNode *, Matcher *)>([&](
                          const ForallNode *n, Matcher *m) {
                    if (n->parallel_unit == PARALLEL_UNIT::GPU_THREAD) {
                      vector<Expr> bounds = relGraph.deriveIterBounds(forall.getIndexVar(), definedIndexVarsMatched,
                                                                      underivedBounds, indexVarToExprMap, iterators);
                      blockSize = ir::Sub::make(bounds[1], bounds[0]);
                    }
                    definedIndexVarsMatched.push_back(n->indexVar);
                  })
            );
          }
          taco_iassert(blockSize.defined());

          if (i == (int) underivedAncestors.size() - 2) {
            std::vector<Expr> args = {
                    posIteratorLevel.getMode().getModePack().getArray(0), // array
                    blockStarts_temporary, // results
                    ir::Literal::zero(posIteratorLevel.getBeginVar().type()), // arrayStart
                    parentSize, // arrayEnd
                    values_per_block, // values_per_block
                    blockSize, // block_size
                    parallelUnitSizes[PARALLEL_UNIT::GPU_BLOCK] // num_blocks
            };
            header.push_back(ir::Assign::make(blockStarts_temporary,
                                              ir::Call::make("taco_binarySearchBeforeBlockLaunch", args,
                                                             getCoordinateVar(underived).type())));
          }
          else {
            std::vector<Expr> args = {
                    posIteratorLevel.getMode().getModePack().getArray(0), // array
                    blockStarts_temporary, // results
                    ir::Literal::zero(posIteratorLevel.getBeginVar().type()), // arrayStart
                    parentSize, // arrayEnd
                    last_block_start_temporary, // targets
                    blockSize, // block_size
                    parallelUnitSizes[PARALLEL_UNIT::GPU_BLOCK] // num_blocks
            };
            header.push_back(ir::Assign::make(blockStarts_temporary,
                                              ir::Call::make("taco_binarySearchIndirectBeforeBlockLaunch", args,
                                                             getCoordinateVar(underived).type())));
          }
          searchForUnderivedStart.push_back(VarDecl::make(posIteratorLevel.getBeginVar(),
                                                          ir::Load::make(blockStarts_temporary,
                                                                         indexVarToExprMap[parallelUnitIndexVars[PARALLEL_UNIT::GPU_BLOCK]])));
          searchForUnderivedStart.push_back(VarDecl::make(posIteratorLevel.getEndVar(),
                                                          ir::Load::make(blockStarts_temporary, ir::Add::make(
                                                                  indexVarToExprMap[parallelUnitIndexVars[PARALLEL_UNIT::GPU_BLOCK]],
                                                                  1))));
          last_block_start_temporary = blockStarts_temporary;
        } else {
          header.push_back(VarDecl::make(posIteratorLevel.getBeginVar(), ir::Literal::zero(posIteratorLevel.getBeginVar().type())));
          header.push_back(VarDecl::make(posIteratorLevel.getEndVar(), parentSize));
        }

        // we do a recovery where we fill in undefined variables with 0's to get start target (just like for vector guards)
        Expr underivedStartTarget;
        if (i == (int) underivedAncestors.size() - 2) {
          std::map<IndexVar, Expr> minChildValues = indexVarToExprMap;
          set<IndexVar> minDefinedIndexVars = definedIndexVars;
          minDefinedIndexVars.erase(forall.getIndexVar());

          for (IndexVar child : relGraph.getFullyDerivedDescendants(posIterator.getIndexVar())) {
            if (!minDefinedIndexVars.count(child)) {
              std::vector<ir::Expr> childBounds = relGraph.deriveIterBounds(child, definedIndexVarsOrdered,
                                                                            underivedBounds,
                                                                            indexVarToExprMap, iterators);
              minChildValues[child] = childBounds[0];

              // recover new parents
              for (const IndexVar &varToRecover : relGraph.newlyRecoverableParents(child, minDefinedIndexVars)) {
                Expr recoveredValue = relGraph.recoverVariable(varToRecover, definedIndexVarsOrdered, underivedBounds,
                                                               minChildValues, iterators);
                taco_iassert(indexVarToExprMap.count(varToRecover));
                searchForUnderivedStart.push_back(VarDecl::make(indexVarToExprMap[varToRecover], recoveredValue));
                minDefinedIndexVars.insert(varToRecover);
                if (varToRecover == posIterator.getIndexVar()) {
                  break;
                }
              }
              minDefinedIndexVars.insert(child);
            }
          }
          underivedStartTarget = indexVarToExprMap[posIterator.getIndexVar()];
        }
        else {
          underivedStartTarget = this->iterators.modeIterator(underivedAncestors[i+1]).getPosVar();
        }

        vector<Expr> binarySearchArgs = {
                posIteratorLevel.getMode().getModePack().getArray(0), // array
                posIteratorLevel.getBeginVar(), // arrayStart
                posIteratorLevel.getEndVar(), // arrayEnd
                underivedStartTarget // target
        };
        Expr posVarUnknown = this->iterators.modeIterator(underivedAncestors[i]).getPosVar();
        searchForUnderivedStart.push_back(ir::VarDecl::make(posVarUnknown,
                                                            ir::Call::make("taco_binarySearchBefore", binarySearchArgs,
                                                                           getCoordinateVar(underivedAncestors[i]).type())));
        Stmt locateCoordVar;
        if (posIteratorLevel.getParent().hasPosIter()) {
          locateCoordVar = ir::VarDecl::make(indexVarToExprMap[underivedAncestors[i]], ir::Load::make(posIteratorLevel.getParent().getMode().getModePack().getArray(1), posVarUnknown));
        }
        else {
          locateCoordVar = ir::VarDecl::make(indexVarToExprMap[underivedAncestors[i]], posVarUnknown);
        }
        searchForUnderivedStart.push_back(locateCoordVar);
      }
    }

    Expr parentPos = this->iterators.modeIterator(underivedAncestors[underivedAncestors.size() - 2]).getPosVar();
    ModeFunction posBounds = posIterator.posBounds(parentPos);
    writeResultCond = ir::Eq::make(ir::Add::make(indexVarToExprMap[posIterator.getIndexVar()], 1), posBounds[1]);

    Stmt loopToTrackUnderiveds; // to track next ancestor
    for (int i = 0; i < (int) underivedAncestors.size() - 1; i++) {
      Expr coordVarUnknown = getCoordinateVar(underivedAncestors[i]);
      Expr posVarKnown = this->iterators.modeIterator(underivedAncestors[i+1]).getPosVar();
      if (i == (int) underivedAncestors.size() - 2) {
        posVarKnown = indexVarToExprMap[posIterator.getIndexVar()];
      }
      Expr posVarUnknown = this->iterators.modeIterator(underivedAncestors[i]).getPosVar();

      Iterator posIteratorLevel = posIterator;
      for (int j = (int) underivedAncestors.size() - 2; j > i; j--) { // take parent of iterator enough times to get correct level
        posIteratorLevel = posIteratorLevel.getParent();
      }

      ModeFunction posBoundsLevel = posIteratorLevel.posBounds(posVarUnknown);
      Expr loopcond = ir::Eq::make(posVarKnown, posBoundsLevel[1]);
      Stmt locateCoordVar;
      if (posIteratorLevel.getParent().hasPosIter()) {
        locateCoordVar = ir::Assign::make(coordVarUnknown, ir::Load::make(posIteratorLevel.getParent().getMode().getModePack().getArray(1), posVarUnknown));
      }
      else {
        locateCoordVar = ir::Assign::make(coordVarUnknown, posVarUnknown);
      }
      Stmt loopBody = ir::Block::make(compoundAssign(posVarUnknown, 1), locateCoordVar, loopToTrackUnderiveds);
      if (posIteratorLevel.getParent().hasPosIter()) { // TODO: if level is unique or not
        loopToTrackUnderiveds = IfThenElse::make(loopcond, loopBody);
      }
      else {
        loopToTrackUnderiveds = While::make(loopcond, loopBody);
      }
    }
    loopsToTrackUnderived.push_back(loopToTrackUnderiveds);
  }

  if (forall.getParallelUnit() != PARALLEL_UNIT::NOT_PARALLEL && forall.getOutputRaceStrategy() == OUTPUT_RACE_STRATEGY::ATOMICS) {
    markAssignsAtomicDepth++;
  }

  Stmt body = lowerForallBody(coordinate, forall.getStmt(),
                              locators, inserters, appenders, reducedAccesses);

  if (forall.getParallelUnit() != PARALLEL_UNIT::NOT_PARALLEL && forall.getOutputRaceStrategy() == OUTPUT_RACE_STRATEGY::ATOMICS) {
    markAssignsAtomicDepth--;
  }

  body = Block::make(recoveryStmt, Block::make(loopsToTrackUnderived), body);

  // Code to write results if using temporary and reset temporary
  if (!whereConsumers.empty() && whereConsumers.back().defined()) {
    Expr temp = tensorVars.find(whereTemps.back())->second;
    Stmt writeResults = Block::make(whereConsumers.back(), ir::Assign::make(temp, ir::Literal::zero(temp.type())));
    // TODO: add reset temp
    body = Block::make(body, IfThenElse::make(writeResultCond, writeResults));
  }

  // Code to append positions
  Stmt posAppend = generateAppendPositions(appenders);

  // Code to compute iteration bounds
  Stmt boundsCompute;
  Expr startBound, endBound;
  if (!relGraph.isUnderived(iterator.getIndexVar())) {
    vector<Expr> bounds = relGraph.deriveIterBounds(iterator.getIndexVar(), definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
    startBound = bounds[0];
    endBound = bounds[1];
  }
  else if (iterator.getParent().isRoot() || iterator.getParent().isUnique()) {
    // E.g. a compressed mode without duplicates
    Expr parentPos = iterator.getParent().getPosVar();
    ModeFunction bounds = iterator.posBounds(parentPos);
    boundsCompute = bounds.compute();
    startBound = bounds[0];
    endBound = bounds[1];
  } else {
    taco_iassert(iterator.isOrdered() && iterator.getParent().isOrdered());
    taco_iassert(iterator.isCompact() && iterator.getParent().isCompact());

    // E.g. a compressed mode with duplicates. Apply iterator chaining
    Expr parentPos = iterator.getParent().getPosVar();
    Expr parentSegend = iterator.getParent().getSegendVar();
    ModeFunction startBounds = iterator.posBounds(parentPos);
    ModeFunction endBounds = iterator.posBounds(ir::Sub::make(parentSegend, 1));
    boundsCompute = Block::make(startBounds.compute(), endBounds.compute());
    startBound = startBounds[0];
    endBound = endBounds[1];
  }

  LoopKind kind = LoopKind::Serial;
  if (forall.getParallelUnit() == PARALLEL_UNIT::CPU_VECTOR && !ignoreVectorize) {
    kind = LoopKind::Vectorized;
  }
  else if (forall.getParallelUnit() != PARALLEL_UNIT::NOT_PARALLEL
           && forall.getOutputRaceStrategy() != OUTPUT_RACE_STRATEGY::PARALLEL_REDUCTION && !ignoreVectorize) {
    kind = LoopKind::Runtime;
  }
  // Loop with preamble and postamble
  return Block::blanks(boundsCompute,
                       Block::make(Block::make(searchForUnderivedStart),
                       For::make(indexVarToExprMap[iterator.getIndexVar()], startBound, endBound, 1,
                                 Block::make(declareCoordinate, body),
                                 kind,
                                 ignoreVectorize ? PARALLEL_UNIT::NOT_PARALLEL : forall.getParallelUnit(), ignoreVectorize ? 0 : forall.getUnrollFactor())),
                       posAppend);

}

Stmt LowererImpl::lowerMergeLattice(MergeLattice lattice, IndexVar coordinateVar,
                                    IndexStmt statement, 
                                    const std::set<Access>& reducedAccesses)
{
  Expr coordinate = getCoordinateVar(coordinateVar);
  vector<Iterator> appenders = filter(lattice.results(),
                                      [](Iterator it){return it.hasAppend();});

  vector<Iterator> mergers = lattice.points()[0].mergers();
  Stmt iteratorVarInits = codeToInitializeIteratorVars(lattice.iterators(), lattice.points()[0].rangers(), mergers, coordinate, coordinateVar);

  // if modeiteratornonmerger then will be declared in codeToInitializeIteratorVars
  auto modeIteratorsNonMergers =
          filter(lattice.points()[0].iterators(), [mergers](Iterator it){
            bool isMerger = find(mergers.begin(), mergers.end(), it) != mergers.end();
            return it.isDimensionIterator() && !isMerger;
          });
  bool resolvedCoordDeclared = !modeIteratorsNonMergers.empty();

  vector<Stmt> mergeLoopsVec;
  for (MergePoint point : lattice.points()) {
    // Each iteration of this loop generates a while loop for one of the merge
    // points in the merge lattice.
    IndexStmt zeroedStmt = zero(statement, getExhaustedAccesses(point,lattice));
    MergeLattice sublattice = lattice.subLattice(point);
    Stmt mergeLoop = lowerMergePoint(sublattice, coordinate, coordinateVar, zeroedStmt, reducedAccesses, resolvedCoordDeclared);
    mergeLoopsVec.push_back(mergeLoop);
  }
  Stmt mergeLoops = Block::make(mergeLoopsVec);

  // Append position to the pos array
  Stmt appendPositions = generateAppendPositions(appenders);

  return Block::blanks(iteratorVarInits,
                       mergeLoops,
                       appendPositions);
}

Stmt LowererImpl::lowerMergePoint(MergeLattice pointLattice,
                                  ir::Expr coordinate, IndexVar coordinateVar, IndexStmt statement,
                                  const std::set<Access>& reducedAccesses, bool resolvedCoordDeclared)
{
  MergePoint point = pointLattice.points().front();

  vector<Iterator> iterators = point.iterators();
  vector<Iterator> mergers = point.mergers();
  vector<Iterator> rangers = point.rangers();
  vector<Iterator> locators = point.locators();

  taco_iassert(iterators.size() > 0);
  taco_iassert(mergers.size() > 0);
  taco_iassert(rangers.size() > 0);

  // Load coordinates from position iterators
  Stmt loadPosIterCoordinates = codeToLoadCoordinatesFromPosIterators(iterators, !resolvedCoordDeclared);

  // Merge iterator coordinate variables
  Stmt resolvedCoordinate = resolveCoordinate(mergers, coordinate, !resolvedCoordDeclared);

  // Locate positions
  Stmt loadLocatorPosVars = declLocatePosVars(locators);

  // Deduplication loops
  auto dupIters = filter(iterators, [](Iterator it){return !it.isUnique() && 
                                                           it.hasPosIter();});
  bool alwaysReduce = (mergers.size() == 1 && mergers[0].hasPosIter());
  Stmt deduplicationLoops = reduceDuplicateCoordinates(coordinate, dupIters, 
                                                       alwaysReduce);

  // One case for each child lattice point lp
  Stmt caseStmts = lowerMergeCases(coordinate, coordinateVar, statement, pointLattice,
                                   reducedAccesses);

  // Increment iterator position variables
  Stmt incIteratorVarStmts = codeToIncIteratorVars(coordinate, coordinateVar, iterators, mergers);

  /// While loop over rangers
  return While::make(checkThatNoneAreExhausted(rangers),
                     Block::make(loadPosIterCoordinates,
                                 resolvedCoordinate,
                                 loadLocatorPosVars,
                                 deduplicationLoops,
                                 caseStmts,
                                 incIteratorVarStmts));
}

Stmt LowererImpl::resolveCoordinate(std::vector<Iterator> mergers, ir::Expr coordinate, bool emitVarDecl) {
  if (mergers.size() == 1) {
    Iterator merger = mergers[0];
    if (merger.hasPosIter()) {
      // Just one position iterator so it is the resolved coordinate
      ModeFunction posAccess = merger.posAccess(merger.getPosVar(),
                                                coordinates(merger));
      Stmt resolution = emitVarDecl ? VarDecl::make(coordinate, posAccess[0]) : Assign::make(coordinate, posAccess[0]);
      return Block::make(posAccess.compute(),
                         resolution);
    }
    else if (merger.hasCoordIter()) {
      taco_not_supported_yet;
      return Stmt();
    }
    else if (merger.isDimensionIterator()) {
      // Just one dimension iterator so resolved coordinate already exist and we
      // do nothing
      return Stmt();
    }
    else {
      taco_ierror << "Unexpected type of single iterator " << merger;
      return Stmt();
    }
  }
  else {
    // Multiple position iterators so the smallest is the resolved coordinate
    if (emitVarDecl) {
      return VarDecl::make(coordinate, Min::make(coordinates(mergers)));
    }
    else {
      return Assign::make(coordinate, Min::make(coordinates(mergers)));
    }
  }
}

Stmt LowererImpl::lowerMergeCases(ir::Expr coordinate, IndexVar coordinateVar, IndexStmt stmt,
                                  MergeLattice lattice,
                                  const std::set<Access>& reducedAccesses)
{
  vector<Stmt> result;

  vector<Iterator> appenders;
  vector<Iterator> inserters;
  tie(appenders, inserters) = splitAppenderAndInserters(lattice.results());

  // Just one iterator so no conditionals
  if (lattice.iterators().size() == 1) {
    Stmt body = lowerForallBody(coordinate, stmt, {}, inserters, 
                                appenders, reducedAccesses);
    result.push_back(body);
  }
  else {
    vector<pair<Expr,Stmt>> cases;
    for (MergePoint point : lattice.points()) {

      // Construct case expression
      vector<Expr> coordComparisons;
      for (Iterator iterator : point.rangers()) {
        if (!(relGraph.isCoordVariable(iterator.getIndexVar()) && relGraph.isDerivedFrom(iterator.getIndexVar(), coordinateVar))) {
          coordComparisons.push_back(Eq::make(iterator.getCoordVar(), coordinate));
        }
      }

      // Construct case body
      IndexStmt zeroedStmt = zero(stmt, getExhaustedAccesses(point, lattice));
      Stmt body = lowerForallBody(coordinate, zeroedStmt, {},
                                  inserters, appenders, reducedAccesses);
      if (coordComparisons.empty()) {
        Stmt body = lowerForallBody(coordinate, stmt, {}, inserters,
                                    appenders, reducedAccesses);
        result.push_back(body);
        break;
      }
      cases.push_back({conjunction(coordComparisons), body});
    }
    result.push_back(Case::make(cases, lattice.exact()));
  }

  return Block::make(result);
}


Stmt LowererImpl::lowerForallBody(Expr coordinate, IndexStmt stmt,
                                  vector<Iterator> locators,
                                  vector<Iterator> inserters,
                                  vector<Iterator> appenders,
                                  const set<Access>& reducedAccesses) {
  Stmt initVals = resizeAndInitValues(appenders, reducedAccesses);

  // Inserter positions
  Stmt declInserterPosVars = declLocatePosVars(inserters);

  // Locate positions
  Stmt declLocatorPosVars = declLocatePosVars(locators);

  if (captureNextLocatePos) {
    capturedLocatePos = Block::make(declInserterPosVars, declLocatorPosVars);
    captureNextLocatePos = false;
  }

  // Code of loop body statement
  Stmt body = lower(stmt);

  // Code to append coordinates
  Stmt appendCoords = appendCoordinate(appenders, coordinate);

  // TODO: Emit code to insert coordinates

  return Block::make(initVals,
                     declInserterPosVars,
                     declLocatorPosVars,
                     body,
                     appendCoords);
}


Stmt LowererImpl::lowerWhere(Where where) {
  TensorVar temporary = where.getTemporary();

  // Declare and initialize the where statement's temporary
  Stmt initializeTemporary = Stmt();
  Stmt freeTemporary = Stmt();
  if (isScalar(temporary.getType())) {
    initializeTemporary = defineScalarVariable(temporary, true);
  }
  else {
    if (generateComputeCode()) {
      Expr values = ir::Var::make(temporary.getName(),
                                  temporary.getType().getDataType(),
                                  true, false);
      taco_iassert(temporary.getType().getOrder() == 1); // TODO
      Dimension temporarySize = temporary.getType().getShape().getDimension(0);
      Expr size;
      if (temporarySize.isFixed()) {
        size = ir::Literal::make(temporarySize.getSize());
      }
      else if (temporarySize.isIndexVarSized()) {
        IndexVar var = temporarySize.getIndexVarSize();
        vector<Expr> bounds = relGraph.deriveIterBounds(var, definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
        size = ir::Sub::make(bounds[1], bounds[0]);
      }
      else {
        taco_ierror; // TODO
      }

      // no decl needed for shared memory
      Stmt decl = Stmt();
      if(isa<Forall>(where.getProducer()) && inParallelLoopDepth == 0) {
        decl = VarDecl::make(values, ir::Literal::make(0));
      }
      Stmt allocate = Allocate::make(values, size);

      Expr p = Var::make("p" + temporary.getName(), Int());
      Stmt zeroInit = Store::make(values, p, ir::Literal::zero(temporary.getType().getDataType()));
      Stmt zeroInitLoop = For::make(p, 0, size, 1, zeroInit, LoopKind::Serial);
      if (markAssignsAtomicDepth > 0) {
        // this temporary was introduced to handle an atomic will be assigned not accumulated TODO:
        zeroInitLoop = Stmt();
      }

      freeTemporary = Free::make(values);

      /// Make a struct object that lowerAssignment and lowerAccess can read
      /// temporary value arrays from.
      TemporaryArrays arrays;
      arrays.values = values;
      this->temporaryArrays.insert({temporary, arrays});

      initializeTemporary = Block::make(decl, allocate, zeroInitLoop);
    }
  }

  match(where.getConsumer(),
        std::function<void(const AssignmentNode*)>([&](const AssignmentNode* op) {
            if (op->lhs.getTensorVar().getOrder() > 0) {
              whereTempsToResult[where.getTemporary()] = (const AccessNode *) op->lhs.ptr;
            }
        })
  );

  Stmt consumer = lower(where.getConsumer());
  whereConsumers.push_back(consumer);
  whereTemps.push_back(where.getTemporary());
  captureNextLocatePos = true;

  // don't apply atomics to producer TODO: mark specific assignments as atomic
  bool restoreAtomicDepth = false;
  if (markAssignsAtomicDepth > 0) {
    markAssignsAtomicDepth--;
    restoreAtomicDepth = true;
  }

  Stmt producer = lower(where.getProducer());

  if (restoreAtomicDepth) {
    markAssignsAtomicDepth++;
  }

  whereConsumers.pop_back();
  whereTemps.pop_back();
  whereTempsToResult.erase(where.getTemporary());
  return Block::make(initializeTemporary, producer, capturedLocatePos, consumer, freeTemporary);
}


Stmt LowererImpl::lowerSequence(Sequence sequence) {
  Stmt definition = lower(sequence.getDefinition());
  Stmt mutation = lower(sequence.getMutation());
  return Block::make(definition, mutation);
}


Stmt LowererImpl::lowerMulti(Multi multi) {
  Stmt stmt1 = lower(multi.getStmt1());
  Stmt stmt2 = lower(multi.getStmt2());
  return Block::make(stmt1, stmt2);
}

Stmt LowererImpl::lowerSuchThat(SuchThat suchThat) {
  Stmt stmt = lower(suchThat.getStmt());
  return Block::make(stmt);
}


Expr LowererImpl::lowerAccess(Access access) {
  TensorVar var = access.getTensorVar();

  if (isScalar(var.getType())) {
    return getTensorVar(var);
  }

  return getIterators(access).back().isUnique()
         ? Load::make(getValuesArray(var), generateValueLocExpr(access))
         : getReducedValueVar(access);
}


Expr LowererImpl::lowerLiteral(Literal literal) {
  switch (literal.getDataType().getKind()) {
    case Datatype::Bool:
      return ir::Literal::make(literal.getVal<bool>());
    case Datatype::UInt8:
      return ir::Literal::make((unsigned long long)literal.getVal<uint8_t>());
    case Datatype::UInt16:
      return ir::Literal::make((unsigned long long)literal.getVal<uint16_t>());
    case Datatype::UInt32:
      return ir::Literal::make((unsigned long long)literal.getVal<uint32_t>());
    case Datatype::UInt64:
      return ir::Literal::make((unsigned long long)literal.getVal<uint64_t>());
    case Datatype::UInt128:
      taco_not_supported_yet;
      break;
    case Datatype::Int8:
      return ir::Literal::make((int)literal.getVal<int8_t>());
    case Datatype::Int16:
      return ir::Literal::make((int)literal.getVal<int16_t>());
    case Datatype::Int32:
      return ir::Literal::make((int)literal.getVal<int32_t>());
    case Datatype::Int64:
      return ir::Literal::make((long long)literal.getVal<int64_t>());
    case Datatype::Int128:
      taco_not_supported_yet;
      break;
    case Datatype::Float32:
      return ir::Literal::make(literal.getVal<float>());
    case Datatype::Float64:
      return ir::Literal::make(literal.getVal<double>());
    case Datatype::Complex64:
      return ir::Literal::make(literal.getVal<std::complex<float>>());
    case Datatype::Complex128:
      return ir::Literal::make(literal.getVal<std::complex<double>>());
    case Datatype::Undefined:
      taco_unreachable;
      break;
  }
  return ir::Expr();
}


Expr LowererImpl::lowerNeg(Neg neg) {
  return ir::Neg::make(lower(neg.getA()));
}


Expr LowererImpl::lowerAdd(Add add) {
  Expr a = lower(add.getA());
  Expr b = lower(add.getB());
  return (add.getDataType().getKind() == Datatype::Bool)
         ? ir::Or::make(a, b) : ir::Add::make(a, b);
}


Expr LowererImpl::lowerSub(Sub sub) {
  return ir::Sub::make(lower(sub.getA()), lower(sub.getB()));
}


Expr LowererImpl::lowerMul(Mul mul) {
  Expr a = lower(mul.getA());
  Expr b = lower(mul.getB());
  return (mul.getDataType().getKind() == Datatype::Bool)
         ? ir::And::make(a, b) : ir::Mul::make(a, b);
}


Expr LowererImpl::lowerDiv(Div div) {
  return ir::Div::make(lower(div.getA()), lower(div.getB()));
}


Expr LowererImpl::lowerSqrt(Sqrt sqrt) {
  return ir::Sqrt::make(lower(sqrt.getA()));
}


Expr LowererImpl::lowerCast(Cast cast) {
  return ir::Cast::make(lower(cast.getA()), cast.getDataType());
}


Expr LowererImpl::lowerCallIntrinsic(CallIntrinsic call) {
  std::vector<Expr> args;
  for (auto& arg : call.getArgs()) {
    args.push_back(lower(arg));
  }
  return call.getFunc().lower(args);
}


Stmt LowererImpl::lower(IndexStmt stmt) {
  return visitor->lower(stmt);
}


Expr LowererImpl::lower(IndexExpr expr) {
  return visitor->lower(expr);
}


bool LowererImpl::generateAssembleCode() const {
  return this->assemble;
}


bool LowererImpl::generateComputeCode() const {
  return this->compute;
}


Expr LowererImpl::getTensorVar(TensorVar tensorVar) const {
  taco_iassert(util::contains(this->tensorVars, tensorVar)) << tensorVar;
  return this->tensorVars.at(tensorVar);
}


Expr LowererImpl::getCapacityVar(Expr tensor) const {
  taco_iassert(util::contains(this->capacityVars, tensor)) << tensor;
  return this->capacityVars.at(tensor);
}


ir::Expr LowererImpl::getValuesArray(TensorVar var) const
{
  return (util::contains(temporaryArrays, var))
         ? temporaryArrays.at(var).values
         : GetProperty::make(getTensorVar(var), TensorProperty::Values);
}


Expr LowererImpl::getDimension(IndexVar indexVar) const {
  taco_iassert(util::contains(this->dimensions, indexVar)) << indexVar;
  return this->dimensions.at(indexVar);
}


std::vector<Iterator> LowererImpl::getIterators(Access access) const {
  vector<Iterator> result;
  TensorVar tensor = access.getTensorVar();
  for (int i = 0; i < tensor.getOrder(); i++) {
    int mode = tensor.getFormat().getModeOrdering()[i];
    result.push_back(iterators.levelIterator(ModeAccess(access, mode+1)));
  }
  return result;
}


set<Access> LowererImpl::getExhaustedAccesses(MergePoint point,
                                              MergeLattice lattice) const
{
  set<Access> exhaustedAccesses;
  for (auto& iterator : lattice.exhausted(point)) {
    exhaustedAccesses.insert(iterators.modeAccess(iterator).getAccess());
  }
  return exhaustedAccesses;
}


Expr LowererImpl::getReducedValueVar(Access access) const {
  return this->reducedValueVars.at(access);
}


Expr LowererImpl::getCoordinateVar(IndexVar indexVar) const {
  return this->iterators.modeIterator(indexVar).getCoordVar();
}


Expr LowererImpl::getCoordinateVar(Iterator iterator) const {
  if (iterator.isDimensionIterator()) {
    return iterator.getCoordVar();
  }
  return this->getCoordinateVar(iterator.getIndexVar());
}


vector<Expr> LowererImpl::coordinates(Iterator iterator) const {
  taco_iassert(iterator.defined());

  vector<Expr> coords;
  do {
    coords.push_back(getCoordinateVar(iterator));
    iterator = iterator.getParent();
  } while (!iterator.isRoot());
  auto reverse = util::reverse(coords);
  return vector<Expr>(reverse.begin(), reverse.end());
}

vector<Expr> LowererImpl::coordinates(vector<Iterator> iterators)
{
  taco_iassert(all(iterators, [](Iterator iter){ return iter.defined(); }));
  vector<Expr> result;
  for (auto& iterator : iterators) {
    result.push_back(iterator.getCoordVar());
  }
  return result;
}


/// Returns true iff a result mode is assembled by inserting a sparse set of 
/// result coordinates (e.g., compressed to dense).
static 
bool hasSparseInserts(const std::vector<Iterator>& resultIterators,
                      const std::multimap<IndexVar, Iterator>& inputIterators) {
  for (const auto& resultIterator : resultIterators) {
    if (resultIterator.hasInsert()) {
      const auto indexVar = resultIterator.getIndexVar();
      const auto accessedInputs = inputIterators.equal_range(indexVar);
      for (auto inputIterator = accessedInputs.first; 
           inputIterator != accessedInputs.second; ++inputIterator) {
        if (!inputIterator->second.isFull()) {
          return true;
        }
      }
    }
  }
  return false;
}


Stmt LowererImpl::initResultArrays(vector<Access> writes, 
                                   vector<Access> reads,
                                   set<Access> reducedAccesses) {
  multimap<IndexVar, Iterator> readIterators;
  for (auto& read : reads) {
    for (auto& readIterator : getIterators(read)) {
      readIterators.insert({readIterator.getIndexVar(), readIterator});
    }
  }

  std::vector<Stmt> result;
  for (auto& write : writes) {
    if (write.getTensorVar().getOrder() == 0) continue;

    std::vector<Stmt> initArrays;

    const auto iterators = getIterators(write);
    taco_iassert(!iterators.empty());

    Expr tensor = getTensorVar(write.getTensorVar());
    Expr valuesArr = GetProperty::make(tensor, TensorProperty::Values);

    Expr parentSize = 1;
    if (generateAssembleCode()) {
      for (const auto& iterator : iterators) {
        Expr size;
        Stmt init;
        // Initialize data structures for storing levels
        if (iterator.hasAppend()) {
          size = 0;
          init = iterator.getAppendInitLevel(parentSize, size);
        } else if (iterator.hasInsert()) {
          size = simplify(ir::Mul::make(parentSize, iterator.getWidth()));
          init = iterator.getInsertInitLevel(parentSize, size);
        } else {
          taco_ierror << "Write iterator supports neither append nor insert";
        }
        initArrays.push_back(init);

        // Declare position variable of append modes that are not above a 
        // branchless mode (if mode below is branchless, then can share same 
        // position variable)
        if (iterator.hasAppend() && (iterator.isLeaf() || 
            !iterator.getChild().isBranchless())) {
          initArrays.push_back(VarDecl::make(iterator.getPosVar(), 0));
        }

        parentSize = size;
      }

      // Pre-allocate memory for the value array if computing while assembling
      if (generateComputeCode()) {
        taco_iassert(!iterators.empty());
        
        Expr capacityVar = getCapacityVar(tensor);
        Expr allocSize = isValue(parentSize, 0) 
                         ? DEFAULT_ALLOC_SIZE : parentSize;
        initArrays.push_back(VarDecl::make(capacityVar, allocSize));
        initArrays.push_back(Allocate::make(valuesArr, capacityVar));
      }

      taco_iassert(!initArrays.empty());
      result.push_back(Block::make(initArrays));
    }
    else if (generateComputeCode()) {
      Iterator lastAppendIterator;
      // Compute size of values array
      for (auto& iterator : iterators) {
        if (iterator.hasAppend()) {
          lastAppendIterator = iterator;
          parentSize = iterator.getSize(parentSize);
        } else if (iterator.hasInsert()) {
          parentSize = ir::Mul::make(parentSize, iterator.getWidth());
        } else {
          taco_ierror << "Write iterator supports neither append nor insert";
        }
        parentSize = simplify(parentSize);
      }

      // Declare position variable for the last append level
      if (lastAppendIterator.defined()) {
        result.push_back(VarDecl::make(lastAppendIterator.getPosVar(), 0));
      }
    }

    if (generateComputeCode() && iterators.back().hasInsert() && 
        !isValue(parentSize, 0) && 
        (hasSparseInserts(iterators, readIterators) || 
         util::contains(reducedAccesses, write))) {
      // Zero-initialize values array if size statically known and might not 
      // assign to every element in values array during compute
      Expr size = generateAssembleCode() ? getCapacityVar(tensor) : parentSize;
      result.push_back(zeroInitValues(tensor, 0, size));
    }
  }
  return result.empty() ? Stmt() : Block::blanks(result);
}


ir::Stmt LowererImpl::finalizeResultArrays(std::vector<Access> writes) {
  if (!generateAssembleCode()) {
    return Stmt();
  }

  std::vector<Stmt> result;
  for (auto& write : writes) {
    if (write.getTensorVar().getOrder() == 0) continue;

    const auto iterators = getIterators(write);
    taco_iassert(!iterators.empty());
      
    Expr parentSize = 1;
    for (const auto& iterator : iterators) {
      Expr size;
      Stmt finalize;
      // Post-process data structures for storing levels
      if (iterator.hasAppend()) {
        size = iterator.getPosVar();
        finalize = iterator.getAppendFinalizeLevel(parentSize, size);
      } else if (iterator.hasInsert()) {
        size = simplify(ir::Mul::make(parentSize, iterator.getWidth()));
        finalize = iterator.getInsertFinalizeLevel(parentSize, size);
      } else {
        taco_ierror << "Write iterator supports neither append nor insert";
      }
      result.push_back(finalize);
      parentSize = size;
    }

    if (!generateComputeCode()) {
      // Allocate memory for values array after assembly if not also computing
      Expr tensor = getTensorVar(write.getTensorVar());
      Expr valuesArr = GetProperty::make(tensor, TensorProperty::Values);
      result.push_back(Allocate::make(valuesArr, parentSize));
    }
  }
  return result.empty() ? Stmt() : Block::blanks(result);
}

Stmt LowererImpl::defineScalarVariable(TensorVar var, bool zero) {
  Datatype type = var.getType().getDataType();
  Expr varValueIR = Var::make(var.getName() + "_val", type, false, false);
  Expr init = (zero) ? ir::Literal::zero(type)
                     : Load::make(GetProperty::make(tensorVars.at(var),
                                                    TensorProperty::Values));
  tensorVars.find(var)->second = varValueIR;
  return VarDecl::make(varValueIR, init);
}

static
vector<Iterator> getIteratorsFrom(IndexVar var, 
                                  const vector<Iterator>& iterators) {
  vector<Iterator> result;
  bool found = false;
  for (Iterator iterator : iterators) {
    if (var == iterator.getIndexVar()) found = true;
    if (found) {
      result.push_back(iterator);
    }
  }
  return result;
}


Stmt LowererImpl::initResultArrays(IndexVar var, vector<Access> writes, 
                                   vector<Access> reads,
                                   set<Access> reducedAccesses) {
  if (!generateAssembleCode()) {
    return Stmt();
  }

  multimap<IndexVar, Iterator> readIterators;
  for (auto& read : reads) {
    for (auto& readIterator : getIteratorsFrom(var, getIterators(read))) {
      readIterators.insert({readIterator.getIndexVar(), readIterator});
    }
  }

  vector<Stmt> result;
  for (auto& write : writes) {
    Expr tensor = getTensorVar(write.getTensorVar());
    Expr values = GetProperty::make(tensor, TensorProperty::Values);

    vector<Iterator> iterators = getIteratorsFrom(var, getIterators(write));

    if (iterators.empty()) {
      continue;
    }

    Iterator resultIterator = iterators.front();

    // Initialize begin var
    if (resultIterator.hasAppend() && !resultIterator.isBranchless()) {
      Expr begin = resultIterator.getBeginVar();
      result.push_back(VarDecl::make(begin, resultIterator.getPosVar()));
    }

    const bool isTopLevel = (iterators.size() == write.getIndexVars().size());
    if (resultIterator.getParent().hasAppend() || isTopLevel) {
      Expr resultParentPos = resultIterator.getParent().getPosVar();
      Expr resultParentPosNext = simplify(ir::Add::make(resultParentPos, 1));
      Expr initBegin = resultParentPos;
      Expr initEnd = resultParentPosNext;
      Expr stride = 1;

      Iterator initIterator;
      for (Iterator iterator : iterators) {
        if (!iterator.hasInsert()) {
          initIterator = iterator;
          break;
        }

        stride = simplify(ir::Mul::make(stride, iterator.getWidth()));
        initBegin = simplify(ir::Mul::make(resultParentPos, stride));
        initEnd = simplify(ir::Mul::make(resultParentPosNext, stride));

        // Initialize data structures for storing insert mode
        result.push_back(iterator.getInsertInitCoords(initBegin, initEnd));
      }

      if (initIterator.defined()) {
        // Initialize data structures for storing edges of next append mode
        taco_iassert(initIterator.hasAppend());
        result.push_back(initIterator.getAppendInitEdges(initBegin, initEnd));
      } else if (generateComputeCode() && !isTopLevel) {
        if (isa<ir::Mul>(stride)) {
          Expr strideVar = Var::make(util::toString(tensor) + "_stride", Int());
          result.push_back(VarDecl::make(strideVar, stride));
          stride = strideVar;
        } 

        // Resize values array if not large enough
        Expr capacityVar = getCapacityVar(tensor);
        Expr size = simplify(ir::Mul::make(resultParentPosNext, stride));
        result.push_back(atLeastDoubleSizeIfFull(values, capacityVar, size));

        if (hasSparseInserts(iterators, readIterators) || 
            util::contains(reducedAccesses, write)) {
          // Zero-initialize values array if might not assign to every element 
          // in values array during compute
          result.push_back(zeroInitValues(tensor, resultParentPos, stride));
        }
      }
    }
  }
  return result.empty() ? Stmt() : Block::make(result);
}


Stmt LowererImpl::resizeAndInitValues(const std::vector<Iterator>& appenders,
                                      const std::set<Access>& reducedAccesses) {
  if (!generateComputeCode()) {
    return Stmt();
  }

  std::function<Expr(Access)> getTensor = [&](Access access) {
    return getTensorVar(access.getTensorVar());
  };
  const auto reducedTensors = util::map(reducedAccesses, getTensor);

  std::vector<Stmt> result;

  for (auto& appender : appenders) {
    if (!appender.isLeaf()) {
      continue;
    }

    Expr tensor = appender.getTensor(); 
    Expr values = GetProperty::make(tensor, TensorProperty::Values);
    Expr capacity = getCapacityVar(appender.getTensor());
    Expr pos = appender.getIteratorVar();

    if (generateAssembleCode()) {
      result.push_back(doubleSizeIfFull(values, capacity, pos));
    }

    if (util::contains(reducedTensors, tensor)) {
      Expr zero = ir::Literal::zero(tensor.type());
      result.push_back(Store::make(values, pos, zero));
    }
  }

  return result.empty() ? Stmt() : Block::make(result);
}


Stmt LowererImpl::zeroInitValues(Expr tensor, Expr begin, Expr size) {
  Expr lower = simplify(ir::Mul::make(begin, size));
  Expr upper = simplify(ir::Mul::make(ir::Add::make(begin, 1), size));
  Expr p = Var::make("p" + util::toString(tensor), Int());
  Expr values = GetProperty::make(tensor, TensorProperty::Values);
  Stmt zeroInit = Store::make(values, p, ir::Literal::zero(tensor.type()));
  LoopKind parallel = (isa<ir::Literal>(size) && 
                       to<ir::Literal>(size)->getIntValue() < (1 << 10))
                      ? LoopKind::Serial : LoopKind::Static_Chunked;
//  if (should_use_CUDA_codegen() &&) {
//    return ir::VarDecl::make(ir::Var::make("status", Int()),
//                                    ir::Call::make("cudaMemset", {values, ir::Literal::make(0, Int()), ir::Mul::make(ir::Sub::make(upper, lower), ir::Literal::make(values.type().getNumBytes()))}, Int()));
//  }
  return For::make(p, lower, upper, 1, zeroInit, parallel);
}


Stmt LowererImpl::declLocatePosVars(vector<Iterator> locators) {
  vector<Stmt> result;
  for (Iterator& locator : locators) {
    accessibleIterators.insert(locator);

    bool doLocate = true;
    for (Iterator ancestorIterator = locator.getParent();
         !ancestorIterator.isRoot() && ancestorIterator.hasLocate();
         ancestorIterator = ancestorIterator.getParent()) {
      if (!accessibleIterators.contains(ancestorIterator)) {
        doLocate = false;
      }
    }

    if (doLocate) {
      Iterator locateIterator = locator;
      if (locateIterator.hasPosIter()) {
        taco_iassert(!relGraph.isUnderived(locateIterator.getIndexVar()));
        continue; // these will be recovered with separate procedure
      }
      do {
        ModeFunction locate = locateIterator.locate(coordinates(locateIterator));
        taco_iassert(isValue(locate.getResults()[1], true));
        Stmt declarePosVar = VarDecl::make(locateIterator.getPosVar(),
                                           locate.getResults()[0]);
        result.push_back(declarePosVar);

        if (locateIterator.isLeaf()) {
          break;
        }
        
        locateIterator = locateIterator.getChild();
      } while (accessibleIterators.contains(locateIterator));
    }
  }
  return result.empty() ? Stmt() : Block::make(result);
}


Stmt LowererImpl::reduceDuplicateCoordinates(Expr coordinate, 
                                             vector<Iterator> iterators,
                                             bool alwaysReduce) {
  vector<Stmt> result;
  for (Iterator& iterator : iterators) {
    taco_iassert(!iterator.isUnique() && iterator.hasPosIter());

    Access access = this->iterators.modeAccess(iterator).getAccess();
    Expr iterVar = iterator.getIteratorVar();
    Expr segendVar = iterator.getSegendVar();
    Expr reducedVal = iterator.isLeaf() ? getReducedValueVar(access) : Expr();
    Expr tensorVar = getTensorVar(access.getTensorVar());
    Expr tensorVals = GetProperty::make(tensorVar, TensorProperty::Values);

    // Initialize variable storing reduced component value.
    if (reducedVal.defined()) {
      Expr reducedValInit = alwaysReduce 
                          ? Load::make(tensorVals, iterVar)
                          : ir::Literal::zero(reducedVal.type());
      result.push_back(VarDecl::make(reducedVal, reducedValInit));
    }

    if (iterator.isLeaf()) {
      // If iterator is over bottommost coordinate hierarchy level and will 
      // always advance (i.e., not merging with another iterator), then we don't 
      // need a separate segend variable.
      segendVar = iterVar;
      if (alwaysReduce) {
        result.push_back(compoundAssign(segendVar, 1));
      }
    } else {
      Expr segendInit = alwaysReduce ? ir::Add::make(iterVar, 1) : iterVar;
      result.push_back(VarDecl::make(segendVar, segendInit));
    } 
    
    vector<Stmt> dedupStmts;
    if (reducedVal.defined()) {
      Expr partialVal = Load::make(tensorVals, segendVar);
      dedupStmts.push_back(compoundAssign(reducedVal, partialVal));
    }
    dedupStmts.push_back(compoundAssign(segendVar, 1));
    Stmt dedupBody = Block::make(dedupStmts);

    ModeFunction posAccess = iterator.posAccess(segendVar, 
                                                coordinates(iterator));
    // TODO: Support access functions that perform additional computations 
    //       and/or might access invalid positions.
    taco_iassert(!posAccess.compute().defined());
    taco_iassert(to<ir::Literal>(posAccess.getResults()[1])->getBoolValue());
    Expr nextCoord = posAccess.getResults()[0];
    Expr withinBounds = Lt::make(segendVar, iterator.getEndVar());
    Expr isDuplicate = Eq::make(posAccess.getResults()[0], coordinate);
    result.push_back(While::make(And::make(withinBounds, isDuplicate),
                                 Block::make(dedupStmts)));
  }
  return result.empty() ? Stmt() : Block::make(result);
}

Stmt LowererImpl::codeToInitializeIteratorVar(Iterator iterator, vector<Iterator> iterators, vector<Iterator> rangers, vector<Iterator> mergers, Expr coordinate, IndexVar coordinateVar) {
  vector<Stmt> result;
  taco_iassert(iterator.hasPosIter() || iterator.hasCoordIter() ||
               iterator.isDimensionIterator());

  Expr iterVar = iterator.getIteratorVar();
  Expr endVar = iterator.getEndVar();
  if (iterator.hasPosIter()) {
    Expr parentPos = iterator.getParent().getPosVar();
    if (iterator.getParent().isRoot() || iterator.getParent().isUnique()) {
      // E.g. a compressed mode without duplicates
      ModeFunction bounds = iterator.posBounds(parentPos);
      result.push_back(bounds.compute());
      // if has a coordinate ranger then need to binary search
      if (any(rangers,
              [](Iterator it){ return it.isDimensionIterator(); })) {

        Expr binarySearchTarget = relGraph.deriveCoordBounds(definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, this->iterators)[coordinateVar][0];
        if (binarySearchTarget != underivedBounds[coordinateVar][0]) {
          result.push_back(VarDecl::make(iterator.getBeginVar(), binarySearchTarget));

          vector<Expr> binarySearchArgs = {
                  iterator.getMode().getModePack().getArray(1), // array
                  bounds[0], // arrayStart
                  bounds[1], // arrayEnd
                  iterator.getBeginVar() // target
          };
          result.push_back(
                  VarDecl::make(iterVar, Call::make("taco_binarySearchAfter", binarySearchArgs, iterVar.type())));
        }
        else {
          result.push_back(VarDecl::make(iterVar, bounds[0]));
        }
      }
      else {
        result.push_back(VarDecl::make(iterVar, bounds[0]));
      }

      result.push_back(VarDecl::make(endVar, bounds[1]));
    } else {
      taco_iassert(iterator.isOrdered() && iterator.getParent().isOrdered());
      taco_iassert(iterator.isCompact() && iterator.getParent().isCompact());

      // E.g. a compressed mode with duplicates. Apply iterator chaining
      Expr parentSegend = iterator.getParent().getSegendVar();
      ModeFunction startBounds = iterator.posBounds(parentPos);
      ModeFunction endBounds = iterator.posBounds(ir::Sub::make(parentSegend, 1));
      result.push_back(startBounds.compute());
      result.push_back(VarDecl::make(iterVar, startBounds[0]));
      result.push_back(endBounds.compute());
      result.push_back(VarDecl::make(endVar, endBounds[1]));
    }
  }
  else if (iterator.hasCoordIter()) {
    // E.g. a hasmap mode
    vector<Expr> coords = coordinates(iterator);
    coords.erase(coords.begin());
    ModeFunction bounds = iterator.coordBounds(coords);
    result.push_back(bounds.compute());
    result.push_back(VarDecl::make(iterVar, bounds[0]));
    result.push_back(VarDecl::make(endVar, bounds[1]));
  }
  else if (iterator.isDimensionIterator()) {
    // A dimension
    // If a merger then initialize to 0
    // If not then get first coord value like doing normal merge

    // If derived then need to recoverchild from this coord value
    bool isMerger = find(mergers.begin(), mergers.end(), iterator) != mergers.end();
    if (isMerger) {
      Expr coord = coordinates(vector<Iterator>({iterator}))[0];
      result.push_back(VarDecl::make(coord, 0));
    }
    else {
      result.push_back(codeToLoadCoordinatesFromPosIterators(iterators, true));

      Stmt stmt = resolveCoordinate(mergers, coordinate, true);
      taco_iassert(stmt != Stmt());
      result.push_back(stmt);
      result.push_back(codeToRecoverDerivedIndexVar(coordinateVar, iterator.getIndexVar(), true));

      // emit bound for ranger too
      vector<Expr> startBounds;
      vector<Expr> endBounds;
      for (Iterator merger : mergers) {
        ModeFunction coordBounds = merger.coordBounds(merger.getParent().getPosVar());
        startBounds.push_back(coordBounds[0]);
        endBounds.push_back(coordBounds[1]);
      }
      //TODO: maybe needed after split reorder? underivedBounds[coordinateVar] = {ir::Max::make(startBounds), ir::Min::make(endBounds)};
      Stmt end_decl = VarDecl::make(iterator.getEndVar(), relGraph.deriveIterBounds(iterator.getIndexVar(), definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, this->iterators)[1]);
      result.push_back(end_decl);
    }
  }
  return result.empty() ? Stmt() : Block::make(result);
}

Stmt LowererImpl::codeToInitializeIteratorVars(vector<Iterator> iterators, vector<Iterator> rangers, vector<Iterator> mergers, Expr coordinate, IndexVar coordinateVar) {
  vector<Stmt> results;
  // initialize mergers first (can't depend on initializing rangers)
  for (Iterator iterator : mergers) {
    results.push_back(codeToInitializeIteratorVar(iterator, iterators, rangers, mergers, coordinate, coordinateVar));
  }

  for (Iterator iterator : rangers) {
      if (find(mergers.begin(), mergers.end(), iterator) == mergers.end()) {
        results.push_back(codeToInitializeIteratorVar(iterator, iterators, rangers, mergers, coordinate, coordinateVar));
      }
  }
  return results.empty() ? Stmt() : Block::make(results);
}

Stmt LowererImpl::codeToRecoverDerivedIndexVar(IndexVar underived, IndexVar indexVar, bool emitVarDecl) {
  if(underived != indexVar) {
    // iterator indexVar must be derived from coordinateVar
    std::vector<IndexVar> underivedAncestors = relGraph.getUnderivedAncestors(indexVar);
    taco_iassert(find(underivedAncestors.begin(), underivedAncestors.end(), underived) != underivedAncestors.end());

    vector<Stmt> recoverySteps;
    for (const IndexVar& varToRecover : relGraph.derivationPath(underived, indexVar)) {
      if(varToRecover == underived) continue;
      recoverySteps.push_back(relGraph.recoverChild(varToRecover, indexVarToExprMap, emitVarDecl, iterators));
    }
    return Block::make(recoverySteps);
  }
  return Stmt();
}

Stmt LowererImpl::codeToIncIteratorVars(Expr coordinate, IndexVar coordinateVar, vector<Iterator> iterators, vector<Iterator> mergers) {
  if (iterators.size() == 1) {
    Expr ivar = iterators[0].getIteratorVar();

    if (iterators[0].isUnique()) {
      return compoundAssign(ivar, 1); 
    }

    // If iterator is over bottommost coordinate hierarchy level with 
    // duplicates and iterator will always advance (i.e., not merging with 
    // another iterator), then deduplication loop will take care of 
    // incrementing iterator variable.
    return iterators[0].isLeaf() 
           ? Stmt()
           : Assign::make(ivar, iterators[0].getSegendVar());
  }

  vector<Stmt> result;

  // We emit the level iterators before the mode iterator because the coordinate
  // of the mode iterator is used to conditionally advance the level iterators.

  auto levelIterators =
      filter(iterators, [](Iterator it){return !it.isDimensionIterator();});
  for (auto& iterator : levelIterators) {
    Expr ivar = iterator.getIteratorVar();
    if (iterator.isUnique()) {
      Expr increment = iterator.isFull()
                     ? 1
                     : ir::Cast::make(Eq::make(iterator.getCoordVar(), 
                                               coordinate),
                                      ivar.type());
      result.push_back(compoundAssign(ivar, increment));
    } else if (!iterator.isLeaf()) {
      result.push_back(Assign::make(ivar, iterator.getSegendVar()));
    }
  }

  auto modeIterators =
      filter(iterators, [](Iterator it){return it.isDimensionIterator();});
  for (auto& iterator : modeIterators) {
    bool isMerger = find(mergers.begin(), mergers.end(), iterator) != mergers.end();
    if (isMerger) {
      Expr ivar = iterator.getIteratorVar();
      result.push_back(compoundAssign(ivar, 1));
    }
    else {
      result.push_back(codeToLoadCoordinatesFromPosIterators(iterators, false));
      Stmt stmt = resolveCoordinate(mergers, coordinate, false);
      taco_iassert(stmt != Stmt());
      result.push_back(stmt);
      result.push_back(codeToRecoverDerivedIndexVar(coordinateVar, iterator.getIndexVar(), false));
    }
  }

  return Block::make(result);
}

Stmt LowererImpl::codeToLoadCoordinatesFromPosIterators(vector<Iterator> iterators, bool declVars) {
  // Load coordinates from position iterators
  Stmt loadPosIterCoordinates;
  if (iterators.size() > 1) {
    vector<Stmt> loadPosIterCoordinateStmts;
    auto posIters = filter(iterators, [](Iterator it){return it.hasPosIter();});
    for (auto& posIter : posIters) {
      taco_tassert(posIter.hasPosIter());
      ModeFunction posAccess = posIter.posAccess(posIter.getPosVar(),
                                                 coordinates(posIter));
      loadPosIterCoordinateStmts.push_back(posAccess.compute());
      if (declVars) {
        loadPosIterCoordinateStmts.push_back(VarDecl::make(posIter.getCoordVar(),
                                                           posAccess[0]));
      }
      else {
        loadPosIterCoordinateStmts.push_back(Assign::make(posIter.getCoordVar(),
                                                          posAccess[0]));
      }
    }
    loadPosIterCoordinates = Block::make(loadPosIterCoordinateStmts);
  }
  return loadPosIterCoordinates;
}


static
bool isLastAppender(Iterator iter) {
  taco_iassert(iter.hasAppend());
  while (!iter.isLeaf()) {
    iter = iter.getChild();
    if (iter.hasAppend()) {
      return false;
    }
  }
  return true;
}


Stmt LowererImpl::appendCoordinate(vector<Iterator> appenders, Expr coord) {
  vector<Stmt> result;
  for (auto& appender : appenders) {
    Expr pos = appender.getPosVar();
    Iterator appenderChild = appender.getChild();

    if (appenderChild.defined() && appenderChild.isBranchless()) {
      // Already emitted assembly code for current level when handling 
      // branchless child level, so don't emit code again.
      continue;
    }

    vector<Stmt> appendStmts;

    if (generateAssembleCode()) {
      appendStmts.push_back(appender.getAppendCoord(pos, coord));
      while (!appender.isRoot() && appender.isBranchless()) {
        // Need to append result coordinate to parent level as well if child 
        // level is branchless (so child coordinates will have unique parents).
        appender = appender.getParent();
        if (!appender.isRoot()) {
          taco_iassert(appender.hasAppend()) << "Parent level of branchless, "
              << "append-capable level must also be append-capable";
          taco_iassert(!appender.isUnique()) << "Need to be able to insert " 
              << "duplicate coordinates to level, but level is declared unique";

          Expr coord = getCoordinateVar(appender);
          appendStmts.push_back(appender.getAppendCoord(pos, coord));
        }
      }
    } 
    
    if (generateAssembleCode() || isLastAppender(appender)) {
      appendStmts.push_back(compoundAssign(pos, 1));

      Stmt appendCode = Block::make(appendStmts);
      if (appenderChild.defined() && appenderChild.hasAppend()) {
        // Emit guard to avoid appending empty slices to result.
        // TODO: Users should be able to configure whether to append zeroes.
        Expr shouldAppend = Lt::make(appenderChild.getBeginVar(), 
                                     appenderChild.getPosVar());
        appendCode = IfThenElse::make(shouldAppend, appendCode);
      }
      result.push_back(appendCode);
    }
  }
  return result.empty() ? Stmt() : Block::make(result);
}


Stmt LowererImpl::generateAppendPositions(vector<Iterator> appenders) {
  vector<Stmt> result;
  if (generateAssembleCode()) {
    for (Iterator appender : appenders) {
      if (!appender.isBranchless()) {
        Expr pos = [](Iterator appender) {
          // Get the position variable associated with the appender. If a mode 
          // is above a branchless mode, then the two modes can share the same 
          // position variable.
          while (!appender.isLeaf() && appender.getChild().isBranchless()) {
            appender = appender.getChild();
          }
          return appender.getPosVar();
        }(appender);
        Expr beginPos = appender.getBeginVar();
        Expr parentPos = appender.getParent().getPosVar();
        result.push_back(appender.getAppendEdges(parentPos, beginPos, pos));
      }
    }
  }
  return result.empty() ? Stmt() : Block::make(result);
}


Expr LowererImpl::generateValueLocExpr(Access access) const {
  if (isScalar(access.getTensorVar().getType())) {
    return ir::Literal::make(0);
  }
  Iterator it = getIterators(access).back();

  if (!access.getIndexVars().empty() && util::contains(indexVarToExprMap, access.getIndexVars().front()) && !it.hasPosIter() && access.getIndexVars().front() == it.getIndexVar()) {
    return indexVarToExprMap.at(access.getIndexVars().front());
  }

  return it.getPosVar();
}


Expr LowererImpl::checkThatNoneAreExhausted(std::vector<Iterator> iterators)
{
  taco_iassert(!iterators.empty());
  if (iterators.size() == 1 && iterators[0].isFull()) {
    std::vector<ir::Expr> bounds = relGraph.deriveIterBounds(iterators[0].getIndexVar(), definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, this->iterators);
    Expr guards = Lt::make(iterators[0].getIteratorVar(), bounds[1]);
    if (bounds[0] != ir::Literal::make(0)) {
      guards = And::make(guards, Gte::make(iterators[0].getIteratorVar(), bounds[0]));
    }
    return guards;
  }

  vector<Expr> result;
  for (const auto& iterator : iterators) {
    taco_iassert(!iterator.isFull()) << iterator
        << " - full iterators do not need to partake in merge loop bounds";
    Expr iterUnexhausted = Lt::make(iterator.getIteratorVar(),
                                    iterator.getEndVar());
    result.push_back(iterUnexhausted);
  }

  return (!result.empty())
         ? conjunction(result)
         : Lt::make(iterators[0].getIteratorVar(), iterators[0].getEndVar());
}

}
