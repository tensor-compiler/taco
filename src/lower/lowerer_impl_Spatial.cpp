#include <taco/lower/mode_format_compressed.h>
#include "taco/lower/lowerer_impl_dataflow.h"
#include "taco/lower/lowerer_impl_Spatial.h"

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
#include "taco/ir/workspace_rewriter.h"
#include "taco/ir/ir_rewriter.h"
#include "taco/spatial.h"

using namespace std;
using namespace taco::ir;
using taco::util::combine;

namespace taco {

class LowererImplSpatial::Visitor : public IndexNotationVisitorStrict {
public:
  Visitor(LowererImplSpatial *impl) : impl(impl) {}

  Stmt lower(IndexStmt stmt) {
    this->stmt = Stmt();
    impl->getAccessibleIterators().scope();
    IndexStmtVisitorStrict::visit(stmt);
    impl->getAccessibleIterators().unscope();
    return this->stmt;
  }

  Expr lower(IndexExpr expr) {
    this->expr = Expr();
    IndexExprVisitorStrict::visit(expr);
    return this->expr;
  }

private:
  LowererImplSpatial *impl;
  Expr expr;
  Stmt stmt;
  using IndexNotationVisitorStrict::visit;

  void visit(const AssignmentNode *node) { stmt = impl->lowerAssignment(node); }

  void visit(const YieldNode *node) { stmt = impl->lowerYield(node); }

  void visit(const ForallNode *node) { stmt = impl->lowerForall(node); }

  void visit(const WhereNode *node) { stmt = impl->lowerWhere(node); }

  void visit(const MultiNode *node) { stmt = impl->lowerMulti(node); }

  void visit(const SuchThatNode *node) { stmt = impl->lowerSuchThat(node); }

  void visit(const SequenceNode *node) { stmt = impl->lowerSequence(node); }

  void visit(const AssembleNode *node) { stmt = impl->lowerAssemble(node); }

  void visit(const AccessNode *node) { expr = impl->lowerAccess(node); }

  void visit(const LiteralNode *node) { expr = impl->lowerLiteral(node); }

  void visit(const NegNode *node) { expr = impl->lowerNeg(node); }

  void visit(const AddNode *node) { expr = impl->lowerAdd(node); }

  void visit(const SubNode *node) { expr = impl->lowerSub(node); }

  void visit(const MulNode *node) { expr = impl->lowerMul(node); }

  void visit(const DivNode *node) { expr = impl->lowerDiv(node); }

  void visit(const SqrtNode *node) { expr = impl->lowerSqrt(node); }

  void visit(const CastNode *node) { expr = impl->lowerCast(node); }

  void visit(const CallIntrinsicNode *node) { expr = impl->lowerCallIntrinsic(node); }

  void visit(const ReductionNode *node) {
    taco_ierror << "Reduction nodes not supported in concrete index notation";
  }
};

LowererImplSpatial::LowererImplSpatial() : visitor(new Visitor(this)) {
}

Stmt LowererImplSpatial::lowerAssignment(Assignment assignment) {
  TensorVar result = assignment.getLhs().getTensorVar();

  if (generateComputeCode()) {
    // First pass of the concrete index notation to see if sparse DRAM accesses need to be hoisted
    Stmt hoistedAccesses = Stmt();
    if (hasSparseDRAMAccesses(assignment.getRhs())) {
      hoistedAccesses = hoistSparseDRAMAccesses(assignment.getRhs());
    }
    Expr var = getTensorVar(result);
    Expr rhs = lower(assignment.getRhs());

    // Assignment to scalar variables.
    if (isScalar(result.getType())) {
      if (!assignment.getOperator().defined()) {
        return Block::blanks(hoistedAccesses, Assign::make(var, rhs, false, getAtomicParallelUnit()));
        // TODO: we don't need to mark all assigns/stores just when scattering/reducing
      } else {
        taco_iassert(isa<taco::Add>(assignment.getOperator()));
        return Block::blanks(hoistedAccesses, compoundAssign(var, rhs, false, getAtomicParallelUnit()));
      }
    }
      // Assignments to tensor variables (non-scalar).
    else {

      Expr values = getValuesArray(result);
      Expr loc = generateValueLocExpr(assignment.getLhs());

      Stmt computeStmt;
      if (!assignment.getOperator().defined()) {
        if (result.getMemoryLocation() == MemoryLocation::SpatialDRAM) {
          computeStmt = MemStore::make(values, rhs, loc, ir::Literal::zero(result.getType().getDataType()));
        } else if (isa<Access>(assignment.getRhs()) &&
                   to<Access>(assignment.getRhs()).getTensorVar().getMemoryLocation() == MemoryLocation::SpatialDRAM) {
          computeStmt = MemLoad::make(values, rhs, loc, ir::Literal::zero(result.getType().getDataType()));
        } else {
          // [Olivia] TODO: see if SpatialReg is correct for RHS
          computeStmt = Store::make(values, loc, rhs, result.getMemoryLocation(), MemoryLocation::SpatialReg, false,
                                    getAtomicParallelUnit());
        }
      } else {
        computeStmt = compoundStore(values, loc, rhs, result.getMemoryLocation(), MemoryLocation::SpatialReg, false,
                                    getAtomicParallelUnit());
      }
      taco_iassert(computeStmt.defined());
      return Block::blanks(hoistedAccesses, computeStmt);
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

bool LowererImplSpatial::hasSparseDRAMAccesses(IndexExpr expression) {
  bool hasSparseDRAMAccess = false;
  match(expression,
        std::function<void(const AccessNode *)>([&](const AccessNode *op) {
          TensorVar var = Access(op).getTensorVar();
          if (var.getMemoryLocation() == MemoryLocation::SpatialSparseDRAM ||
              var.getMemoryLocation() == MemoryLocation::SpatialSparseDRAMFalse)
            hasSparseDRAMAccess = true;
        })
  );

  return hasSparseDRAMAccess;
}

Stmt LowererImplSpatial::hoistSparseDRAMAccesses(IndexExpr expression) {
  vector<Stmt> hoistedStmts;
  match(expression,
        std::function<void(const AccessNode *)>([&](const AccessNode *op) {
          TensorVar var = Access(op).getTensorVar();
          if (!isScalar(var.getType()) && (var.getMemoryLocation() == MemoryLocation::SpatialSparseDRAM ||
                                           var.getMemoryLocation() == MemoryLocation::SpatialSparseDRAMFalse)) {
            auto vals = getValuesArray(var);
            auto loc = generateValueLocExpr(Access(op));

            auto tempVals = ir::Var::make(op->tensorVar.getName() + "_vals_raw", vals.type());
            auto tempValsDecl = ir::VarDecl::make(tempVals, ir::Load::make(vals, loc, var.getMemoryLocation()));
            hoistedStmts.push_back(tempValsDecl);
            sparseDRAMAccessMap.insert({var, tempVals});

          }
        })
  );
  return Block::make(hoistedStmts);
}

Expr LowererImplSpatial::lowerAccess(Access access) {
  TensorVar var = access.getTensorVar();

  if (isScalar(var.getType())) {
    return getTensorVar(var);
  }

  if ((var.getMemoryLocation() == MemoryLocation::SpatialSparseDRAM ||
       var.getMemoryLocation() == MemoryLocation::SpatialSparseDRAMFalse)
      && sparseDRAMAccessMap.find(var) != sparseDRAMAccessMap.end()) {
    return sparseDRAMAccessMap.at(var);
  }

  if (hoistedAccessVars.count(var) > 0) {
    return hoistedAccessVars.at(var);
  }

  return getIterators(access).back().isUnique()
         ? Load::make(getValuesArray(var), generateValueLocExpr(access), var.getMemoryLocation())
         : getReducedValueVar(access);
}

Stmt LowererImplSpatial::lowerForall(Forall forall) {
  bool hasExactBound = provGraph.hasExactBound(forall.getIndexVar());
  bool forallNeedsUnderivedGuards = !hasExactBound && emitUnderivedGuards;
  if (!ignoreVectorize && forallNeedsUnderivedGuards &&
      (forall.getParallelUnit() == ParallelUnit::CPUVector ||
       forall.getUnrollFactor() > 0)) {
    return lowerForallCloned(forall);
  }

  if (forall.getParallelUnit() != ParallelUnit::NotParallel) {
    inParallelLoopDepth++;
  }

  // Recover any available parents that were not recoverable previously
  vector<Stmt> recoverySteps;
  for (const IndexVar &varToRecover : provGraph.newlyRecoverableParents(forall.getIndexVar(), definedIndexVars)) {
    // place pos guard
    if (forallNeedsUnderivedGuards && provGraph.isCoordVariable(varToRecover) &&
        provGraph.getChildren(varToRecover).size() == 1 &&
        provGraph.isPosVariable(provGraph.getChildren(varToRecover)[0])) {
      IndexVar posVar = provGraph.getChildren(varToRecover)[0];
      std::vector<ir::Expr> iterBounds = provGraph.deriveIterBounds(posVar, definedIndexVarsOrdered, underivedBounds,
                                                                    indexVarToExprMap, iterators);

      Expr minGuard = Lt::make(indexVarToExprMap[posVar], iterBounds[0]);
      Expr maxGuard = Gte::make(indexVarToExprMap[posVar], iterBounds[1]);
      Expr guardCondition = Or::make(minGuard, maxGuard);
      if (isa<ir::Literal>(ir::simplify(iterBounds[0])) &&
          ir::simplify(iterBounds[0]).as<ir::Literal>()->equalsScalar(0)) {
        guardCondition = maxGuard;
      }
      ir::Stmt guard = ir::IfThenElse::make(guardCondition, ir::Continue::make());
      recoverySteps.push_back(guard);
    }

    Expr recoveredValue = provGraph.recoverVariable(varToRecover, definedIndexVarsOrdered, underivedBounds,
                                                    indexVarToExprMap, iterators);
    taco_iassert(indexVarToExprMap.count(varToRecover));
    recoverySteps.push_back(VarDecl::make(indexVarToExprMap[varToRecover], recoveredValue));

    // After we've recovered this index variable, some iterators are now
    // accessible for use when declaring locator access variables. So, generate
    // the accessors for those locator variables as part of the recovery process.
    // This is necessary after a fuse transformation, for example: If we fuse
    // two index variables (i, j) into f, then after we've generated the loop for
    // f, all locate accessors for i and j are now available for use.
    std::vector<Iterator> itersForVar;
    for (auto &iters : iterators.levelIterators()) {
      // Collect all level iterators that have locate and iterate over
      // the recovered index variable.
      if (iters.second.getIndexVar() == varToRecover && iters.second.hasLocate()) {
        itersForVar.push_back(iters.second);
      }
    }
    // Finally, declare all of the collected iterators' position access variables.
    recoverySteps.push_back(this->declLocatePosVars(itersForVar));

    // place underived guard
    std::vector<ir::Expr> iterBounds = provGraph.deriveIterBounds(varToRecover, definedIndexVarsOrdered,
                                                                  underivedBounds, indexVarToExprMap, iterators);
    if (forallNeedsUnderivedGuards && underivedBounds.count(varToRecover) &&
        !provGraph.hasPosDescendant(varToRecover)) {

      // FIXME: [Olivia] Check this with someone
      // Removed underived guard if indexVar is bounded is divisible by its split child indexVar
      vector<IndexVar> children = provGraph.getChildren(varToRecover);
      bool hasDirectDivBound = false;
      std::vector<ir::Expr> iterBoundsInner = provGraph.deriveIterBounds(forall.getIndexVar(), definedIndexVarsOrdered,
                                                                         underivedBounds, indexVarToExprMap, iterators);

      for (auto &c: children) {
        if (provGraph.hasExactBound(c) && provGraph.derivationPath(varToRecover, c).size() == 2) {
          std::vector<ir::Expr> iterBoundsUnderivedChild = provGraph.deriveIterBounds(c, definedIndexVarsOrdered,
                                                                                      underivedBounds,
                                                                                      indexVarToExprMap, iterators);
          if (iterBoundsUnderivedChild[1].as<ir::Literal>()->getValue<int>() %
              iterBoundsInner[1].as<ir::Literal>()->getValue<int>() == 0)
            hasDirectDivBound = true;
          break;
        }
      }
      if (!hasDirectDivBound) {
        Stmt guard = IfThenElse::make(Gte::make(indexVarToExprMap[varToRecover], underivedBounds[varToRecover][1]),
                                      Continue::make());
        recoverySteps.push_back(guard);
      }
    }

    // If this index variable was divided into multiple equal chunks, then we
    // must add an extra guard to make sure that further scheduling operations
    // on descendent index variables exceed the bounds of each equal portion of
    // the loop. For a concrete example, consider a loop of size 10 that is divided
    // into two equal components -- 5 and 5. If the loop is then transformed
    // with .split(..., 3), each inner chunk of 5 will be split into chunks of
    // 3. Without an extra guard, the second chunk of 3 in the first group of 5
    // may attempt to perform an iteration for the second group of 5, which is
    // incorrect.
    if (this->provGraph.isDivided(varToRecover)) {
      // Collect the children iteration variables.
      auto children = this->provGraph.getChildren(varToRecover);
      auto outer = children[0];
      auto inner = children[1];
      // Find the iteration bounds of the inner variable -- that is the size
      // that the outer loop was broken into.
      auto bounds = this->provGraph.deriveIterBounds(inner, definedIndexVarsOrdered, underivedBounds, indexVarToExprMap,
                                                     iterators);
      // Use the difference between the bounds to find the size of the loop.
      auto dimLen = ir::Sub::make(bounds[1], bounds[0]);
      // For a variable f divided into into f1 and f2, the guard ensures that
      // for iteration f, f should be within f1 * dimLen and (f1 + 1) * dimLen.
      auto guard = ir::Gte::make(this->indexVarToExprMap[varToRecover],
                                 ir::Mul::make(ir::Add::make(this->indexVarToExprMap[outer], 1), dimLen));
      recoverySteps.push_back(IfThenElse::make(guard, ir::Continue::make()));
    }
  }
  Stmt recoveryStmt = Block::make(recoverySteps);

  taco_iassert(!definedIndexVars.count(forall.getIndexVar())) << forall.getIndexVar();
  definedIndexVars.insert(forall.getIndexVar());
  definedIndexVarsOrdered.push_back(forall.getIndexVar());

  if (forall.getParallelUnit() != ParallelUnit::NotParallel && forall.getParallelUnit() != ParallelUnit::Spatial) {
    taco_iassert(!parallelUnitSizes.count(forall.getParallelUnit()));
    taco_iassert(!parallelUnitIndexVars.count(forall.getParallelUnit()));
    parallelUnitIndexVars[forall.getParallelUnit()] = forall.getIndexVar();
    vector<Expr> bounds = provGraph.deriveIterBounds(forall.getIndexVar(), definedIndexVarsOrdered, underivedBounds,
                                                     indexVarToExprMap, iterators);
    parallelUnitSizes[forall.getParallelUnit()] = ir::Sub::make(bounds[1], bounds[0]);
  }

  MergeLattice lattice = MergeLattice::make(forall, iterators, provGraph, definedIndexVars, whereTempsToResult);
  vector<Access> resultAccesses;
  set<Access> reducedAccesses;
  std::tie(resultAccesses, reducedAccesses) = getResultAccesses(forall);

  // Pre-allocate/initialize memory of value arrays that are full below this
  // loops index variable
  Stmt preInitValues = initResultArrays(forall.getIndexVar(), resultAccesses,
                                        getArgumentAccesses(forall),
                                        reducedAccesses);

  // Emit temporary initialization if forall is sequential and leads to a where statement
  vector<Stmt> temporaryValuesInitFree = {Stmt(), Stmt()};
  auto temp = temporaryInitialization.find(forall);
  if (temp != temporaryInitialization.end() && forall.getParallelUnit() == ParallelUnit::NotParallel &&
      !isScalar(temp->second.getTemporary().getType()))
    temporaryValuesInitFree = codeToInitializeTemporary(temp->second);

  Stmt loops;
  // Emit a loop that iterates over over a single iterator (optimization)
  if (lattice.iterators().size() == 1 && lattice.iterators()[0].isUnique()) {
    taco_iassert(lattice.points().size() == 1);

    MergePoint point = lattice.points()[0];
    Iterator iterator = lattice.iterators()[0];

    if (forall != outerForall) {
      Forall subforall;
      if (isa<Forall>(forall.getStmt())) {
        subforall = forall.getStmt().as<Forall>();

//      } else if (isa<Where>(forall.getStmt()) && isa<Forall>(forall.getStmt().as<Where>().getConsumer())) {
//        subforall = forall.getStmt().as<Where>().getConsumer().as<Forall>();

      } else if (isa<Where>(forall.getStmt()) && isa<Forall>(forall.getStmt().as<Where>().getProducer())) {
        subforall = forall.getStmt().as<Where>().getProducer().as<Forall>();
      }

      if (subforall.defined()) {
        auto sublattice = MergeLattice::make(subforall, iterators, provGraph,
                                             definedIndexVars, whereTempsToResult);
        Iterator subIterator = sublattice.iterators()[0];

        if (subIterator.getMode().defined()) {
          auto posArr = subIterator.getMode().getModePack().getArray(0);

          auto parentVar = subIterator.getParent().getPosVar();
          hoistedPosArr[forall] = posArr;
          hoistedArrIterator[forall] = subIterator.getParent();

          const GetProperty* posArrGP = posArr.as<GetProperty>();
          gpToVarMap[posArr] = ir::Var::make(posArrGP->name, posArrGP->type);

        }
      }
    }

    if (forall != outerForall) {
      Assignment assignment;
      if (isa<Assignment>(forall.getStmt())) {
        assignment = forall.getStmt().as<Assignment>();
      } else if (isa<Where>(forall.getStmt()) && isa<Assignment>(forall.getStmt().as<Where>().getProducer())) {
        assignment = forall.getStmt().as<Where>().getProducer().as<Assignment>();
      }




        auto lattice = MergeLattice::make(forall, iterators, provGraph,
                                             definedIndexVars, whereTempsToResult);
        Iterator iterator = lattice.iterators()[0];

        if (iterator.getMode().defined()) {
          auto crdArr = iterator.getMode().getModePack().getArray(1);

          auto parentVar = iterator.getParent().getPosVar();
          hoistedCrdArr[forall] = crdArr;
          hoistedArrIterator[forall] = iterator;

          const GetProperty* crdArrGP = crdArr.as<GetProperty>();
          gpToVarMap[crdArr] = ir::Var::make(crdArrGP->name, crdArrGP->type);

          if (assignment.defined()) {
            Expr valArr;
            for (auto& tv : tensorVars) {
              if (tv.second == crdArr.as<GetProperty>()->tensor)
                valArr = getValuesArray(tv.first);
            }
            taco_iassert(valArr.defined()) << "Values array must be in tensorVars map";

            hoistedValArr[forall] = valArr;
            gpToVarMap[valArr] = ir::Var::make(valArr.as<GetProperty>()->name, valArr.as<GetProperty>()->type);
          }
        }


      }
    //}

    vector<Iterator> locators = point.locators();
    vector<Iterator> appenders;
    vector<Iterator> inserters;
    tie(appenders, inserters) = splitAppenderAndInserters(point.results());
    std::vector<IndexVar> underivedAncestors = provGraph.getUnderivedAncestors(iterator.getIndexVar());
    IndexVar posDescendant;
    bool hasPosDescendant = false;
    if (!underivedAncestors.empty()) {
      hasPosDescendant = provGraph.getPosIteratorFullyDerivedDescendant(underivedAncestors[0], &posDescendant);
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

    // For now, this only works when consuming a single workspace.
    //bool canAccelWithSparseIteration = inParallelLoopDepth == 0 && provGraph.isFullyDerived(iterator.getIndexVar()) &&
    //                                   iterator.isDimensionIterator() && locators.size() == 1;
    bool canAccelWithSparseIteration =
      provGraph.isFullyDerived(iterator.getIndexVar()) &&
      iterator.isDimensionIterator() && locators.size() == 1 && forall.getParallelUnit() == ParallelUnit::NotParallel;
    if (canAccelWithSparseIteration) {
      bool indexListsExist = false;
      // We are iterating over a dimension and locating into a temporary with a tracker to keep indices. Instead, we
      // can just iterate over the indices and locate into the dense workspace.
      for (auto it = tensorVars.begin(); it != tensorVars.end(); ++it) {
        if (it->second == locators[0].getTensor() && util::contains(tempToIndexList, it->first)) {
          indexListsExist = true;
          break;
        }
      }
      canAccelWithSparseIteration &= indexListsExist;
    }

    if (!isWhereProducer && hasPosDescendant && underivedAncestors.size() > 1 &&
        provGraph.isPosVariable(iterator.getIndexVar()) && posDescendant == forall.getIndexVar()) {
      loops = lowerForallFusedPosition(forall, iterator, locators,
                                       inserters, appenders, reducedAccesses, recoveryStmt);
    }


      // Emit dimension coordinate iteration loop
    else if (iterator.isDimensionIterator()) {
      loops = lowerForallDimension(forall, iterator, point.locators(),
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
    std::vector<IndexVar> underivedAncestors = provGraph.getUnderivedAncestors(forall.getIndexVar());
    taco_iassert(underivedAncestors.size() == 1); // TODO: add support for fused coordinate of pos loop
    loops = lowerDeclarativeSparse(lattice, underivedAncestors[0], forall,
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
  if (forall.getParallelUnit() != ParallelUnit::NotParallel && forall.getParallelUnit() != ParallelUnit::Spatial) {
    inParallelLoopDepth--;
    taco_iassert(parallelUnitSizes.count(forall.getParallelUnit()));
    taco_iassert(parallelUnitIndexVars.count(forall.getParallelUnit()));
    parallelUnitIndexVars.erase(forall.getParallelUnit());
    parallelUnitSizes.erase(forall.getParallelUnit());
  }

  return Block::blanks(preInitValues,
                       temporaryValuesInitFree[0],
                       loops,
                       temporaryValuesInitFree[1]);
}

ir::Expr LowererImplSpatial::getValuesArray(TensorVar var) const {
  return (util::contains(getTemporaryArrays(), var))
         ? getTemporaryArrays().at(var).values
         : GetProperty::make(getTensorVar(var), TensorProperty::Values, 0, var.getOrder(),
                             util::contains(var.getFormat().getModeFormats(), ModeFormat::sparse));
}

vector<Stmt> LowererImplSpatial::codeToInitializeTemporary(Where where) {
  TensorVar temporary = where.getTemporary();

  Stmt freeTemporary = Stmt();
  Stmt initializeTemporary = Stmt();
  if (isScalar(temporary.getType())) {
    initializeTemporary = defineScalarVariable(temporary, true);
  } else {
    if (generateComputeCode()) {

      /// Make a struct object that lowerAssignment and lowerAccess can read
      /// temporary value arrays from.
      TemporaryArrays arrays;

      Expr values = ir::Var::make(temporary.getName(),
                                  temporary.getType().getDataType(),
                                  true, false);

      Expr size = getTemporarySize(where);

      // allocate ws values array
      MemoryLocation tempMemLoc = temporary.getMemoryLocation();
      switch (tempMemLoc) {
        case MemoryLocation::SpatialFIFORetimed:
        case MemoryLocation::SpatialFIFO:
          size = 16;
        default:
          size = ir::Div::make(funcEnvMap.at("nnz_accel_max"), 4);
      }

      Stmt allocate = Allocate::make(values, size, false, Expr(), false, tempMemLoc);

      Expr p = Var::make("p" + temporary.getName(), Int());
      Stmt zeroInit = Store::make(values, p, ir::Literal::zero(temporary.getType().getDataType()));
      Stmt zeroInitLoop = For::make(p, 0, size, 1, funcEnvMap["ip"], zeroInit, LoopKind::Serial);

      // allocate ws indices and dimension arrays
      if (tempMemLoc != MemoryLocation::SpatialDRAM || tempMemLoc != MemoryLocation::SpatialSparseDRAM) {

        vector<Stmt> indArrays;
        auto temporaryModeArrays = getAllTemporaryModeArrays(where);
        for (auto &tempArray : temporaryModeArrays) {
          auto gp = tempArray.as<GetProperty>();
          MemoryLocation memLoc = MemoryLocation::Default;
          if (gp->property == TensorProperty::Indices) {

            switch (tempMemLoc) {
              case MemoryLocation::SpatialFIFO:
                memLoc = (gp->index == 0) ? MemoryLocation::SpatialSRAM : MemoryLocation::SpatialFIFO;
                break;
              case MemoryLocation::SpatialSparseSRAM:
              default:
                memLoc = (gp->index == 0) ? MemoryLocation::SpatialSRAM : MemoryLocation::SpatialSparseSRAM;
                break;
            }

            Expr indices = ir::Var::make(gp->name, gp->type, true, false, false, memLoc);

            Expr indSize = (gp->index == 0) ? ir::Add::make(
              ir::GetProperty::make(gp->tensor, TensorProperty::Dimension, gp->mode - 1), 1)
                                            : funcEnvMap["nnz_accel_max"];
            Stmt allocateInd = Allocate::make(indices, indSize, false, Expr(), false, memLoc);
            indArrays.push_back(allocateInd);

            arrays.indices[gp->name] = indices;
          }
        }

        allocate = Block::make(Block::make(indArrays), allocate);
      }

      arrays.values = values;
      this->insertTemporaryArrays(temporary, arrays);

      freeTemporary = Free::make(values);

      if (getTempNoZeroInit().find(temporary) != getTempNoZeroInit().end())
        initializeTemporary = Block::make(allocate);
      else
        initializeTemporary = Block::make(allocate, zeroInitLoop);

      // Don't zero initialize temporary if there is no reduction across temporary
      if (isa<Forall>(where.getProducer())) {
        Forall forall = to<Forall>(where.getProducer());
        if (isa<Assignment>(forall.getStmt()) && !to<Assignment>(forall.getStmt()).getOperator().defined()) {
          initializeTemporary = Block::make(allocate);
        }
      }

    }
  }

  return {initializeTemporary, freeTemporary};
}

Stmt LowererImplSpatial::lowerDeclarativeSparse(MergeLattice lattice, IndexVar coordinateVar,
                                                Forall forall, IndexStmt statement,
                                                const std::set<Access> &reducedAccesses) {
  Expr coordinate = getCoordinateVar(coordinateVar);
  vector<Iterator> appenders = filter(lattice.results(),
                                      [](Iterator it) { return it.hasAppend(); });

  vector<Iterator> mergers = lattice.points()[0].mergers();

  Stmt iteratorVarInits = codeToInitializeIteratorVars(lattice.iterators(), lattice.points()[0].rangers(), mergers,
                                                       coordinate, coordinateVar);

  // if modeiteratornonmerger then will be declared in codeToInitializeIteratorVars
  auto modeIteratorsNonMergers =
    filter(lattice.points()[0].iterators(), [mergers](Iterator it) {
      bool isMerger = find(mergers.begin(), mergers.end(), it) != mergers.end();
      return it.isDimensionIterator() && !isMerger;
    });
  bool resolvedCoordDeclared = !modeIteratorsNonMergers.empty();

  auto rootPoint = lattice.points()[0];
  // 1) Load from DRAM to FIFO
  map<Iterator, ir::Expr> varMap;
  Stmt memoryTransfer = loadDRAMtoFIFO(statement, rootPoint, varMap);

  // 2) generate BitVectors from FIFO to another FIFO
  map<Iterator, ir::Expr> bvMap;
  Stmt genBitVectors = generateIteratorBitVectors(statement, coordinate, coordinateVar, rootPoint, varMap, bvMap);

  // 3) Calculate pos arrays using scanner
  // An iteration lattice point represents a union when it contains more than 2 iterators and
  // dominates other points (has more than itself as a point in the sublattice).
  bool isUnion = rootPoint.iterators().size() > 1 && lattice.subLattice(rootPoint).points().size() > 1;

  // Append position to the pos array
  Stmt appendPositions = generateIteratorAppendPositions(statement, coordinate, coordinateVar, rootPoint, appenders,
                                                         varMap, isUnion);

  // 4) Calculate add
  Stmt computeLoop = generateIteratorComputeLoop(forall, statement, coordinate, coordinateVar, rootPoint, lattice,
                                                 bvMap,
                                                 reducedAccesses, isUnion);

  return Block::blanks(iteratorVarInits, memoryTransfer, genBitVectors, appendPositions, computeLoop);
}

Stmt LowererImplSpatial::lowerForallDimension(Forall forall, Iterator iterator,
                                              vector<Iterator> locators,
                                              vector<Iterator> inserters,
                                              vector<Iterator> appenders,
                                              set<Access> reducedAccesses,
                                              ir::Stmt recoveryStmt) {
  Expr numChunks = (forall == outerForall) ? funcEnvMap["bp"] :
    (util::contains(innerForalls, forall)) ? funcEnvMap["ip"] :
    (forall.getNumChunks() > 0 && forall.getNumChunks() <= 16) ? (int) forall.getNumChunks() : 1;

  Expr coordinate = getCoordinateVar(forall.getIndexVar());

  if (forall.getParallelUnit() != ParallelUnit::NotParallel &&
      forall.getOutputRaceStrategy() == OutputRaceStrategy::Atomics) {
    markAssignsAtomicDepth++;
    atomicParallelUnit = forall.getParallelUnit();
  }


  parentLoopVar.push_back(coordinate);
  Stmt body = lowerForallBody(coordinate, forall.getStmt(),
                              locators, inserters, appenders, reducedAccesses);

  parentLoopVar.pop_back();

  if (forall.getParallelUnit() != ParallelUnit::NotParallel &&
      forall.getOutputRaceStrategy() == OutputRaceStrategy::Atomics) {
    markAssignsAtomicDepth--;
  }

  // If outer par is > 1, move result allocation into forall
  Stmt allocResultArrs = generateOPResultHoist(forall, body);

  body = Block::make({allocResultArrs, recoveryStmt, body});

  // Emit loop with preamble and postamble
  std::vector<ir::Expr> bounds = provGraph.deriveIterBounds(forall.getIndexVar(), definedIndexVarsOrdered,
                                                            underivedBounds, indexVarToExprMap, iterators);

  Expr appendLoopVar = coordinate;
  if ( parentLoopVar.size() > 0) {
    appendLoopVar =  parentLoopVar.back();
  }
  Stmt posAppend = generateAppendPositionsForallPos(appenders, bounds[1],  appendLoopVar);

  auto startStore = (parentLoopVar.size() > 0) ? parentLoopVar.back() : 1;
  startStore = ir::Mul::make(startStore, bounds[1]);
  Stmt resultStore = generateResultStore(forall, appenders, startStore, bounds[1]);
  posAppend = Block::make(posAppend, resultStore);

  LoopKind kind = LoopKind::Serial;
  if (forall.getParallelUnit() == ParallelUnit::CPUVector && !ignoreVectorize) {
    kind = LoopKind::Vectorized;
  } else if (forall.getParallelUnit() != ParallelUnit::NotParallel
             && forall.getOutputRaceStrategy() != OutputRaceStrategy::ParallelReduction && !ignoreVectorize) {
    kind = LoopKind::Runtime;
  }


  Stmt posTransfers = Block::make();
  if (hoistedPosArr.count(forall) > 0) {
    auto arrIterator = hoistedArrIterator.at(forall);
    auto bounds = coordinateBounds.at(arrIterator.getPosVar());

    auto posArr = hoistedPosArr.at(forall);
    auto posDram = ir::Var::make(posArr.as<GetProperty>()->name + "_dram", posArr.type());
    auto memLoc = MemoryLocation::SpatialSRAM;
    auto allocatePosArr = ir::Allocate::make(posArr, ir::Div::make(funcEnvMap["nnz_accel_max"], 4),
                                             false, Expr(), false, memLoc);
    auto posArrLoad = ir::LoadBulk::make(posDram, get<0>(bounds), ir::Add::make(get<1>(bounds), 1), funcEnvMap["ip"]);
    auto posArrStore = ir::StoreBulk::make(posArr, posArrLoad, memLoc, MemoryLocation::SpatialDRAM);
    posTransfers = Block::make(allocatePosArr, posArrStore);
  }

  vector<Expr> memLoadStart;
  vector<Expr> memLoadEnd;

  for (auto &tns : forall.getCommunicateTensors()) {
    for (auto &locator : locators) {
      if (tensorVars.count(tns) > 0 &&
          locator.getTensor() == tensorVars[tns]) {
        auto coords = coordinates(locator);
        taco_iassert (coords.size() > 0 && coords.back() == coordinate);
        coords.pop_back();
        taco_iassert(coords.size() > 0);

        auto parentVar = coords.back();
        memLoadStart.push_back(ir::Mul::make(parentVar, locator.getWidth()));
        memLoadEnd.push_back(ir::Mul::make(ir::Add::make(parentVar, 1), locator.getWidth()));
      }
    }
  }

  if (forall.getParallelUnit() == ParallelUnit::Spatial &&
      forall.getOutputRaceStrategy() == OutputRaceStrategy::SpatialReduction) {
    if (forallReductions.find(forall) != forallReductions.end() && isa<Assignment>(forall.getStmt())) {
      Assignment forallExpr = to<Assignment>(forall.getStmt());

      Expr reg;
      Stmt regDecl = Stmt();
      if (forallReductions.at(forall).first > 0) {
        reg = Var::make("r_" + forall.getIndexVar().getName() + "_" + forallExpr.getLhs().getTensorVar().getName(),
                        forallExpr.getLhs().getDataType());
        regDecl = VarDecl::make(reg, ir::Literal::zero(reg.type()), MemoryLocation::SpatialReg);
      } else {
        // TODO: If reduction across a non-scalar, need to use a memreduce into a SpatialSRAM
        reg = lower(forallExpr.getLhs());
        // Only declare register for upper-most level if it is a scalar
        if (isScalar(to<Access>(forallExpr.getLhs()).getTensorVar().getType()))
          regDecl = VarDecl::make(reg, ir::Literal::zero(reg.type()), MemoryLocation::SpatialReg);
      }

      Stmt reductionBody = removeEndReduction(body);

      reductionBody = Block::make({recoveryStmt, reductionBody});

      Expr reductionExpr = lower(forallExpr.getRhs());

      // FIXME: reduction can only handle adds for now
      taco_iassert(isa<taco::Add>(forallExpr.getOperator()));

      if (forallExpr.getOperator().defined()) {
        auto reduce = Block::make(posTransfers, regDecl, Reduce::make(coordinate, reg, bounds[0], bounds[1], 1,
                                                        numChunks, Scope::make(reductionBody, reductionExpr),
                                                        true));
        for (int i = 0; i < (int) forall.getCommunicateTensors().size(); i++) {
          if (!hasResult(appenders, forall.getCommunicateTensors()[i])) {
            reduce = generateOPMemLoads(forall, reduce, memLoadStart[i], memLoadEnd[i],
                                        forall.getCommunicateTensors()[i]);
          }
        }
        return reduce;
      }
    }
    if (forallReductions.find(forall) != forallReductions.end() &&
        !provGraph.getParents(forall.getIndexVar()).empty()) {
      Assignment forallExpr = forallReductions.at(forall).second;

      Expr reg;
      Stmt regDecl = Stmt();
      if (forallReductions.at(forall).first > 0) {
        reg = Var::make("r_" + forall.getIndexVar().getName() + "_" + forallExpr.getLhs().getTensorVar().getName(),
                        forallExpr.getLhs().getDataType());
        regDecl = VarDecl::make(reg, ir::Literal::zero(reg.type()), MemoryLocation::SpatialReg);
      } else {
        // TODO: If reduction across a non-scalar, need to use a memreduce into a SpatialSRAM
        reg = lower(forallExpr.getLhs());
        // Only declare register for upper-most level if it is a scalar
        if (isScalar(to<Access>(forallExpr.getLhs()).getTensorVar().getType()))
          regDecl = VarDecl::make(reg, ir::Literal::zero(reg.type()), MemoryLocation::SpatialReg);
      }

      // FIXME: reduction can only handle adds for now
      taco_iassert(isa<taco::Add>(forallExpr.getOperator()));
      auto parentVar = provGraph.getParents(forall.getIndexVar())[0];
      vector<IndexVar> children = provGraph.getChildren(parentVar);

      if (forallExpr.getOperator().defined()) {
        auto reduce = Block::make(posTransfers, regDecl, Reduce::make(coordinate, reg, bounds[0], bounds[1], 1,
                                                        numChunks, body, true));
        for (int i = 0; i < (int) forall.getCommunicateTensors().size(); i++) {
          if (!hasResult(appenders, forall.getCommunicateTensors()[i])) {
            reduce = generateOPMemLoads(forall, reduce, memLoadStart[i], memLoadEnd[i],
                                        forall.getCommunicateTensors()[i]);
          }
        }
        return reduce;
      }
    }
  }

  if (bulkMemTransfer.find(forall) != bulkMemTransfer.end()) {
    auto assignment = bulkMemTransfer.at(forall);
    auto tensorLhs = assignment.getLhs().getTensorVar();
    Expr valuesLhs = getValuesArray(tensorLhs);
    auto tensorRhs = to<Access>(assignment.getRhs()).getTensorVar();
    Expr valuesRhs = getValuesArray(tensorRhs);

    //Stmt vars = lowerForallReductionBody(coordinate, forall.getStmt(),
    //                                     locators, inserters, appenders, reducedAccesses);
    auto locs = lowerForallBulk(forall, coordinate, forall.getStmt(),
                                locators, inserters, appenders, reducedAccesses, recoveryStmt);

    Expr data = LoadBulk::make(valuesRhs, ir::Add::make(get<1>(locs[1]), bounds[0]),
                               ir::Add::make(get<1>(locs[1]), bounds[1]));

    auto block = Block::make(posTransfers, get<0>(locs[0]), StoreBulk::make(valuesLhs,
                                                              data,
                                                              tensorLhs.getMemoryLocation(), tensorRhs.getMemoryLocation()));
    return block;
  }

  auto loop = Block::blanks(posTransfers, For::make(coordinate, bounds[0], bounds[1], 1, numChunks, body,
                                      kind,
                                      ignoreVectorize ? ParallelUnit::NotParallel : forall.getParallelUnit(),
                                      ignoreVectorize ? 0 : forall.getUnrollFactor(), 0),
                            posAppend);

  for (int i = 0; i < (int) forall.getCommunicateTensors().size(); i++) {
    if (!hasResult(appenders, forall.getCommunicateTensors()[i]) && memLoadStart.size() > i ) {
      loop = generateOPMemLoads(forall, loop, memLoadStart[i], memLoadEnd[i], forall.getCommunicateTensors()[i]);
    }
  }

  return loop;
}

Stmt LowererImplSpatial::lowerForallPosition(Forall forall, Iterator iterator,
                                             vector<Iterator> locators,
                                             vector<Iterator> inserters,
                                             vector<Iterator> appenders,
                                             set<Access> reducedAccesses,
                                             ir::Stmt recoveryStmt) {
  Expr numChunks = (forall == outerForall) ? funcEnvMap["bp"] :
                   (util::contains(innerForalls, forall)) ? funcEnvMap["ip"] :
                   (forall.getNumChunks() > 0 && forall.getNumChunks() <= 16) ? (int) forall.getNumChunks() : 1;

  Expr coordinate = getCoordinateVar(forall.getIndexVar());

  Stmt declareCoordinate = Stmt();
  if (getProvGraph().isCoordVariable(forall.getIndexVar())) {
    Expr coordinateArray = iterator.posAccess(iterator.getPosVar(),
                                              coordinates(iterator)).getResults()[0];
    declareCoordinate = VarDecl::make(coordinate, coordinateArray, iterator.getMode().getMemoryLocation());

  }

  auto appenderDecls = declLocatePosVars(appenders);

  // Generate correct coordinate appender
  Stmt appendCoordVar = generateAppendCoordVar(appenders, iterator.getPosVar());

  recoveryStmt = Block::make(appendCoordVar, recoveryStmt);

  if (forall.getParallelUnit() != ParallelUnit::NotParallel &&
      forall.getOutputRaceStrategy() == OutputRaceStrategy::Atomics) {
    markAssignsAtomicDepth++;
  }

  Stmt hoistedTensorAccess = generateHoistedtensorAccess(forall);


  // Code to compute iteration bounds
  Stmt boundsCompute;
  Expr startBound = 0;
  Expr endBound = iterator.getParent().getEndVar();
  Expr parentPos = iterator.getParent().getPosVar();
  if (!getProvGraph().isUnderived(iterator.getIndexVar())) {
    vector<Expr> bounds = getProvGraph().deriveIterBounds(iterator.getIndexVar(), getDefinedIndexVarsOrdered(),
                                                          getUnderivedBounds(), getIndexVarToExprMap(), getIterators());
    startBound = bounds[0];
    endBound = bounds[1];
  } else if (iterator.getParent().isRoot() || iterator.getParent().isUnique()) {
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

  auto loopVar = to<Var>(iterator.getPosVar());

  Expr posAppendVar = endBound;
  vector<Expr> memLoadBounds = {startBound, ir::Sub::make(endBound, startBound)};
  // Spatial needs memory accesses to be described before loop
  Stmt boundsDecl;
  if (!isa<Var>(endBound) && !isa<Var>(startBound)) {
    auto startVar = Var::make(loopVar->name + "_start", startBound.type());
    auto startBoundDecl = VarDecl::make(startVar, startBound);

    auto endVar = Var::make(loopVar->name + "_end", endBound.type());
    auto endBoundDecl = VarDecl::make(endVar, endBound);

    auto lenVar = Var::make(loopVar->name + "_len", endBound.type());
    auto lenVarDecl = VarDecl::make(lenVar, ir::Sub::make(endVar, startVar));

    boundsDecl = Block::make(startBoundDecl, endBoundDecl, lenVarDecl);
    startBound = ir::Literal::zero(startBound.type());
    endBound = lenVar;

    memLoadBounds = {startVar, lenVar};

    posAppendVar = endVar;
  } else if (!isa<Var>(startBound)) {
    //TODO: Remove duplicate code here
    auto startVar = Var::make(loopVar->name + "_start", startBound.type());
    auto startBoundDecl = VarDecl::make(startVar, startBound);
    boundsDecl = startBoundDecl;
    startBound = startVar;

    memLoadBounds = {startBound, ir::Sub::make(endBound, startBound)};
  } else if (!isa<Var>(endBound)) {
    //TODO: Remove duplicate code here
    auto endVar = Var::make(loopVar->name + "_end", endBound.type());
    auto endBoundDecl = VarDecl::make(endVar, endBound);
    boundsDecl = Block::make(boundsDecl, endBoundDecl);
    endBound = endVar;

    posAppendVar = endVar;
    memLoadBounds = {startBound, ir::Sub::make(endBound, startBound)};
  }


  coordinateBounds[iterator.getPosVar()] = {memLoadBounds[0], posAppendVar};

  parentLoopVar.push_back(loopVar);
  Stmt body = lowerForallBody(coordinate, forall.getStmt(),
                              locators, inserters, appenders, reducedAccesses);

  parentLoopVar.pop_back();

  body = Block::make(hoistedTensorAccess, body);

  // If outer par is > 1, move result allocation into forall
  Stmt allocResultArrs = generateOPResultHoist(forall, body);

  // Add in resultVals to beginning of loop
  declareCoordinate = Block::blanks(allocResultArrs, appenderDecls, declareCoordinate);

  // Cleanup GetProperties from resultAccesses above
  if (forall == outerForall && envValMap.at("bp").as<ir::Literal>()->getTypedVal() > 1) {
    // Get all properties from result
    // move result allocation here.. but how?
    for (auto &access : resultTensorAccesses) {
      body = addGPLoadFlag(body, access.getTensorVar(), tensorVars);
    }
  }

  if (forall.getParallelUnit() != ParallelUnit::NotParallel &&
      forall.getOutputRaceStrategy() == OutputRaceStrategy::Atomics) {
    markAssignsAtomicDepth--;
  }

  Stmt posTransfers = Block::make();
  if (hoistedPosArr.count(forall) > 0) {
    auto posArr = hoistedPosArr.at(forall);
    auto posDram = ir::Var::make(posArr.as<GetProperty>()->name + "_dram", posArr.type());
    auto memLoc = MemoryLocation::SpatialSRAM;
    auto allocatePosArr = ir::Allocate::make(posArr, ir::Div::make(funcEnvMap["nnz_accel_max"], 4),
                                             false, Expr(), false, memLoc);
    auto posArrLoad = ir::LoadBulk::make(posDram, memLoadBounds[0], ir::Add::make(posAppendVar, 1), funcEnvMap["ip"]);
    auto posArrStore = ir::StoreBulk::make(posArr, posArrLoad, memLoc, MemoryLocation::SpatialDRAM);
    posTransfers = Block::make(allocatePosArr, posArrStore);
  }

  if (hoistedCrdArr.count(forall) > 0) {
    auto crdArr = hoistedCrdArr.at(forall);
    auto crdDram = ir::Var::make(crdArr.as<GetProperty>()->name + "_dram", crdArr.type());
    auto memLoc = crdArr.as<GetProperty>()->tensor.as<Var>()->memoryLocation;
    Expr size;
    Expr endLoad;
    if (memLoc == MemoryLocation::SpatialFIFO || memLoc == MemoryLocation::SpatialFIFORetimed) {
      size = 16;
      endLoad = memLoadBounds[1];
    } else {
      size = ir::Div::make(funcEnvMap["nnz_accel_max"], 4);
      endLoad = posAppendVar;
    }
    auto allocateCrdArr = ir::Allocate::make(crdArr, size,
                                             false, Expr(), false, memLoc);
    auto crdArrLoad = ir::LoadBulk::make(crdDram, memLoadBounds[0], endLoad, funcEnvMap["ip"]);
    auto crdArrStore = ir::StoreBulk::make(crdArr, crdArrLoad, memLoc, MemoryLocation::SpatialDRAM);
    posTransfers = Block::make(posTransfers, allocateCrdArr, crdArrStore);

    if (hoistedValArr.count(forall) > 0) {
      Expr valArr = hoistedValArr.at(forall);
      taco_iassert(valArr.defined()) << "Values array must be in tensorVars map";

      auto valDram = ir::Var::make(valArr.as<GetProperty>()->name + "_dram", valArr.type());

      auto allocateValArr = ir::Allocate::make(valArr, size,
                                               false, Expr(), false, memLoc);
      auto valArrLoad = ir::LoadBulk::make(valDram, memLoadBounds[0], endLoad, funcEnvMap["ip"]);
      auto valArrStore = ir::StoreBulk::make(valArr, valArrLoad, memLoc, MemoryLocation::SpatialDRAM);
      posTransfers = Block::make(posTransfers, allocateValArr, valArrStore);
    }
  }


  // Code to append positions
  Expr appendLoopVar = loopVar;
  if ( parentLoopVar.size() > 0) {
    appendLoopVar =  parentLoopVar.back();
  }
  Stmt posAppend = generateAppendPositionsForallPos(appenders, posAppendVar, appendLoopVar);

  Stmt resultStore = generateResultStore(forall, appenders, memLoadBounds[0], memLoadBounds[1]);
  posAppend = Block::make(posAppend, resultStore);

  LoopKind kind = LoopKind::Serial;
  if (forall.getParallelUnit() == ParallelUnit::CPUVector && !ignoreVectorize) {
    kind = LoopKind::Vectorized;
  } else if (forall.getParallelUnit() != ParallelUnit::NotParallel
             && forall.getOutputRaceStrategy() != OutputRaceStrategy::ParallelReduction && !ignoreVectorize) {
    kind = LoopKind::Runtime;
  }

  if (forall.getParallelUnit() == ParallelUnit::Spatial &&
      forall.getOutputRaceStrategy() == OutputRaceStrategy::SpatialReduction) {
    if (forallReductions.find(forall) != forallReductions.end() && isa<Assignment>(forall.getStmt())) {
      Assignment forallExpr = to<Assignment>(forall.getStmt());

      Expr reg;
      Stmt regDecl = Stmt();
      if (forallReductions.at(forall).first > 0) {
        reg = Var::make("r_" + forall.getIndexVar().getName() + "_" + forallExpr.getLhs().getTensorVar().getName(),
                        forallExpr.getLhs().getDataType());
        regDecl = VarDecl::make(reg, ir::Literal::zero(reg.type()), MemoryLocation::SpatialReg);
      } else {
        // TODO: If reduction across a non-scalar, need to use a memreduce into a SpatialSRAM
        reg = lower(forallExpr.getLhs());
        // Only declare register for upper-most level if it is a scalar
        if (isScalar(to<Access>(forallExpr.getLhs()).getTensorVar().getType()))
          regDecl = VarDecl::make(reg, ir::Literal::zero(reg.type()), MemoryLocation::SpatialReg);
      }

      Stmt reductionBody = removeEndReduction(body);
      reductionBody = Block::make({recoveryStmt, reductionBody});

      Expr reductionExpr = lower(forallExpr.getRhs());

      taco_iassert(isa<taco::Add>(forallExpr.getOperator())) << "Spatial reductions can only handle additions";

      if (forallExpr.getOperator().defined()) {
        auto reduce = Reduce::make(loopVar, reg, startBound, endBound, 1, numChunks,
                                   Block::make(declareCoordinate,
                                               Scope::make(reductionBody,
                                                           reductionExpr)),
                                   true);
        if (!forall.getCommunicateTensors().empty() && !hasResult(appenders, forall.getCommunicateTensors()[0]))
          reduce = generateOPMemLoads(forall, reduce, memLoadBounds[0], memLoadBounds[1],
                                      forall.getCommunicateTensors()[0]);
        return Block::make(boundsCompute, Block::blanks(boundsDecl, posTransfers, Block::blanks(regDecl, reduce)));
      }
    }
      //else if (forallReductions.find(forall) != forallReductions.end() && !provGraph.getParents(forall.getIndexVar()).empty()) {
    else if (forallReductions.find(forall) != forallReductions.end()) {
      Assignment forallExpr = forallReductions.at(forall).second;

      Expr reg;
      Stmt regDecl = Stmt();
      if (forallReductions.at(forall).first > 0) {
        reg = Var::make("r_" + forall.getIndexVar().getName() + "_" + forallExpr.getLhs().getTensorVar().getName(),
                        forallExpr.getLhs().getDataType());
        regDecl = VarDecl::make(reg, ir::Literal::zero(reg.type()), MemoryLocation::SpatialReg);
      } else {
        // TODO: If reduction across a non-scalar, need to use a memreduce into a SpatialSRAM
        reg = lower(forallExpr.getLhs());
        // Only declare register for upper-most level if it is a scalar
        if (isScalar(to<Access>(forallExpr.getLhs()).getTensorVar().getType()))
          regDecl = VarDecl::make(reg, ir::Literal::zero(reg.type()), MemoryLocation::SpatialReg);
      }

      taco_iassert(isa<taco::Add>(forallExpr.getOperator())) << "Spatial reductions can only handle additions";


      if (forallExpr.getOperator().defined()) {
        auto reduce = Reduce::make(loopVar, reg, startBound, endBound,
                                   1, numChunks, Block::make(declareCoordinate, body), true);
        // Code to generate loads and stores from DRAM if tagged with communicate
        if (!forall.getCommunicateTensors().empty() && !hasResult(appenders, forall.getCommunicateTensors()[0]))
          reduce = generateOPMemLoads(forall, reduce, memLoadBounds[0], memLoadBounds[1],
                                      forall.getCommunicateTensors()[0]);

        return Block::make(boundsCompute, Block::blanks(boundsDecl, posTransfers, Block::blanks(regDecl, reduce)));
      }
    }
  }

  auto loop = For::make(loopVar, startBound, endBound, 1, numChunks,
                        Block::make(declareCoordinate, body),
                        kind,
                        ignoreVectorize ? ParallelUnit::NotParallel : forall.getParallelUnit(),
                        ignoreVectorize ? 0 : forall.getUnrollFactor());
  // Code to generate loads and stores from DRAM if tagged with communicate
  if (!forall.getCommunicateTensors().empty() && !hasResult(appenders, forall.getCommunicateTensors()[0]))
    loop = generateOPMemLoads(forall, loop, memLoadBounds[0], memLoadBounds[1], forall.getCommunicateTensors()[0]);

  // Loop with preamble and postamble
  return Block::blanks(boundsCompute, boundsDecl, Block::blanks(posTransfers,
                       loop), posAppend);

}

Stmt LowererImplSpatial::generateAppendCoordVar(vector<Iterator> appenders, Expr coordinatePosVar) {
  vector<Stmt> appendStmts;
  for (auto &appender : appenders) {
    Expr pos = appender.getPosVar();
    if (generateComputeCode()) {
      auto posDecl = VarDecl::make(pos, coordinatePosVar);
      appendStmts.push_back(posDecl);
    }
  }
  return Block::make(appendStmts);
}

Stmt LowererImplSpatial::generateIteratorComputeLoop(Forall forall, IndexStmt statement, ir::Expr coordinate,
                                                     IndexVar coordinateVar, MergePoint point,
                                                     MergeLattice lattice, map<Iterator, ir::Expr> &varMap,
                                                     const std::set<Access> &reducedAccesses, bool isUnion) {

  taco_iassert(point.iterators().size() <= 2) << "Spatial/Capstan hardware can only handle two sparse iterators"
                                                 "at once. To fix this please try inserting a temporary workspace";

  vector<Iterator> appenders;
  vector<Iterator> inserters;
  tie(appenders, inserters) = splitAppenderAndInserters(lattice.results());

  auto uncompressedCrd = ir::Var::make(coordinate.as<Var>()->name + "_uncomp", coordinate.as<Var>()->type);
  auto compressedCrd = ir::Var::make(coordinate.as<Var>()->name + "_comp", coordinate.as<Var>()->type);

  taco_iassert(indexVartoBitVarMap.find(coordinateVar) != indexVartoBitVarMap.end()) << "Iterator not found in bit "
                                                                                        "variable map defined by global "
                                                                                        "metadata environment";
  auto bitLen = ir::Mul::make(ir::Literal::make(32), indexVartoBitVarMap.at(coordinateVar));

  taco_iassert(funcEnvMap.find("sp") != funcEnvMap.end()) << "Scanner Par is not in global function environment";

  bool isReduction = (forall.getParallelUnit() == ParallelUnit::Spatial &&
                      forall.getOutputRaceStrategy() == OutputRaceStrategy::SpatialReduction);

  bool isInnerLoop = (isa<Assignment>(forall.getStmt()));
  auto parFactor = isInnerLoop ? funcEnvMap.at("ip") : funcEnvMap.at("sp");

  Expr scan = Expr();
  Expr typeCase = Expr();
  if (point.iterators().size() == 1) {
    auto iteratorVar = varMap[point.iterators()[0]];
    auto load = ir::Load::make(iteratorVar, iteratorVar.as<Var>()->memoryLocation);
    // TODO: fix bitcnt and par factor later
    scan = ir::Scan::make(parFactor,
                          bitLen,
                          load, Expr(), isUnion, isReduction);


    typeCase = ir::TypeCase::make({uncompressedCrd, compressedCrd});
  } else {
    auto iterator1 = point.iterators()[0];
    auto iterator2 = point.iterators()[1];
    auto iteratorVar1 = varMap[iterator1];
    auto iteratorVar2 = varMap[iterator2];
    // TODO: fix bitcnt and par factor later
    auto load1 = ir::Load::make(iteratorVar1, iteratorVar2.as<Var>()->memoryLocation);
    auto load2 = ir::Load::make(iteratorVar2, iteratorVar2.as<Var>()->memoryLocation);
    scan = ir::Scan::make(parFactor, bitLen, load1, load2, isUnion, isReduction && isInnerLoop);

    typeCase = ir::TypeCase::make({uncompressedCrd, iterator1.getCoordVar(), compressedCrd, iterator2.getCoordVar()});
    coordinateScanVarsMap[coordinate] = {uncompressedCrd, iterator1.getCoordVar(), compressedCrd,
                                         iterator2.getCoordVar()};
  }

  vector<Stmt> iterVars;
  for (Iterator appender : appenders) {
    Expr pos = [](Iterator appender) {
      // Get the position variable associated with the appender. If a mode
      // is above a branchless mode, then the two modes can share the same
      // position variable.
      while (!appender.isLeaf() && appender.getChild().isBranchless()) {
        appender = appender.getChild();
      }
      return appender.getPosVar();
    }(appender);

    // FIXME: need to get access to pos base
    Stmt iterDecl = ir::VarDecl::make(appender.getIteratorVar(), ir::Add::make(appender.getBeginVar(), compressedCrd));
    iterVars.push_back(iterDecl);
  }

  for (auto &iterator : point.iterators()) {
    auto ternary = ir::Ternary::make(ir::Gte::make(iterator.getCoordVar(), ir::Literal::zero(Int())),
                                     ir::Add::make(iterator.getBeginVar(), iterator.getCoordVar()),
                                     ir::Literal::make(-1, Int()));
    Stmt iterDecl = ir::VarDecl::make(iterator.getIteratorVar(), ternary);
    iterVars.push_back(iterDecl);
  }


  previousIteratorCoord.push_back(coordinate);
  Stmt body = lowerForallBody(coordinate, statement, {}, inserters,
                              appenders, reducedAccesses);
  previousIteratorCoord.pop_back();

  body = Block::make(Block::make(iterVars), body);

  // TODO: check to match on reduction flag. If so, create reduction instead of ForScan
  if (isReduction) {
    if (forallReductions.find(forall) != forallReductions.end() && isa<Assignment>(forall.getStmt())) {
      Assignment forallExpr = to<Assignment>(statement);

      Expr reg;
      Stmt regDecl = Stmt();
      if (forallReductions.at(forall).first > 0) {
        reg = Var::make("r_" + forall.getIndexVar().getName() + "_" + forallExpr.getLhs().getTensorVar().getName(),
                        forallExpr.getLhs().getDataType());
        regDecl = VarDecl::make(reg, ir::Literal::zero(reg.type()), MemoryLocation::SpatialReg);
      } else {
        // TODO: If reduction across a non-scalar, need to use a memreduce into a SpatialSRAM
        reg = lower(forallExpr.getLhs());
        // Only declare register for upper-most level if it is a scalar
        if (isScalar(to<Access>(forallExpr.getLhs()).getTensorVar().getType()))
          regDecl = VarDecl::make(reg, ir::Literal::zero(reg.type()), MemoryLocation::SpatialReg);
      }

      Stmt reductionBody = removeEndReduction(body);

      Expr reductionExpr = lower(forallExpr.getRhs());

      // FIXME: reduction can only handle adds for now
      taco_iassert(isa<taco::Add>(forallExpr.getOperator()));

      if (forallExpr.getOperator().defined()) {
        return Block::make(regDecl, ir::ReduceScan::make(typeCase, reg,
                                                         scan, Scope::make(reductionBody, reductionExpr), true));
      }
    } else if (forallReductions.find(forall) != forallReductions.end()) {
      Assignment forallExpr = forallReductions.at(forall).second;

      Expr reg;
      Stmt regDecl = Stmt();
      if (forallReductions.at(forall).first > 0) {
        reg = Var::make("r_" + forall.getIndexVar().getName() + "_" + forallExpr.getLhs().getTensorVar().getName(),
                        forallExpr.getLhs().getDataType());
        regDecl = VarDecl::make(reg, ir::Literal::zero(reg.type()), MemoryLocation::SpatialReg);
      } else {
        // TODO: If reduction across a non-scalar, need to use a memreduce into a SpatialSRAM
        reg = lower(forallExpr.getLhs());
        // Only declare register for upper-most level if it is a scalar
        if (isScalar(to<Access>(forallExpr.getLhs()).getTensorVar().getType()))
          regDecl = VarDecl::make(reg, ir::Literal::zero(reg.type()), MemoryLocation::SpatialReg);
      }

      taco_iassert(isa<taco::Add>(forallExpr.getOperator())) << "Spatial reductions can only handle additions";

      if (forallExpr.getOperator().defined()) {
        return Block::make(regDecl, ir::ReduceScan::make(typeCase, reg, scan, Scope::make(body), true));
      }
    }
  }
  auto loop = ir::ForScan::make(typeCase, scan, body);

  return loop;
}

Stmt
LowererImplSpatial::generateIteratorAppendPositions(IndexStmt statement, ir::Expr coordinate, IndexVar coordinateVar,
                                                    MergePoint point,
                                                    std::vector<Iterator> appenders, map<Iterator, ir::Expr> &varMap,
                                                    bool isUnion) {
  taco_iassert(point.iterators().size() <= 2) << "Spatial/Capstan hardware can only handle two sparse iterators"
                                                 "at once. To fix this please try inserting a temporary workspace";

  auto uncompressedCrd = ir::Var::make(coordinate.as<Var>()->name + "_uncomp", coordinate.as<Var>()->type);
  auto compressedCrd = ir::Var::make(coordinate.as<Var>()->name + "_comp", coordinate.as<Var>()->type);

  taco_iassert(indexVartoBitVarMap.find(coordinateVar) != indexVartoBitVarMap.end()) << "Iterator not found in bit "
                                                                                        "variable map defined by global "
                                                                                        "metadata environment";
  auto bitLen = ir::Mul::make(ir::Literal::make(32), indexVartoBitVarMap.at(coordinateVar));

  bool isInnerLoop = (isa<Assignment>(statement));
  auto parFactor = isInnerLoop ? funcEnvMap.at("ip") : funcEnvMap.at("sp");

  Expr scan = Expr();
  Expr typeCase = Expr();
  if (point.iterators().size() == 1) {
    auto iteratorVar = varMap[point.iterators()[0]];
    auto load = ir::Load::make(iteratorVar, iteratorVar.as<Var>()->memoryLocation);
    // TODO: fix bitcnt and par factor later
    scan = ir::Scan::make(parFactor, bitLen,
                          load, Expr(), isUnion, true);


    typeCase = ir::TypeCase::make({uncompressedCrd, compressedCrd});
  } else {
    auto iterator1 = point.iterators()[0];
    auto iterator2 = point.iterators()[1];
    auto iteratorVar1 = varMap[iterator1];
    auto iteratorVar2 = varMap[iterator2];
    // TODO: fix bitcnt and par factor later
    auto load1 = ir::Load::make(iteratorVar1, iteratorVar2.as<Var>()->memoryLocation);
    auto load2 = ir::Load::make(iteratorVar2, iteratorVar2.as<Var>()->memoryLocation);
    scan = ir::Scan::make(parFactor, bitLen, load1, load2, isUnion, true);

    typeCase = ir::TypeCase::make({uncompressedCrd, iterator1.getCoordVar(), compressedCrd, iterator2.getCoordVar()});
  }

  // Generate reduction register
  auto reg = ir::Var::make(coordinateVar.getName() + "_r_pos", coordinate.type(), false, false, false,
                           MemoryLocation::SpatialReg);
  auto regDecl = ir::VarDecl::make(reg, ir::Literal::zero(reg.type()), MemoryLocation::SpatialReg);

  // Create reduction that counts up all of the bits in the bitvectors. This will be stored in the reulst Pos array
  // FIXME: weird spatial bug. See if uncompressedCrd is the correct variable here
  auto reductionBody = ir::Add::make(ir::Literal::make(1), ir::Div::make(compressedCrd, ir::Literal::make(10000)));
  auto reduction = ir::ReduceScan::make(typeCase, reg, scan, Stmt(), reductionBody);

  vector<Stmt> result;
  for (Iterator appender : appenders) {

    Expr pos = [](Iterator appender) {
      // Get the position variable associated with the appender. If a mode
      // is above a branchless mode, then the two modes can share the same
      // position variable.
      while (!appender.isLeaf() && appender.getChild().isBranchless()) {
        appender = appender.getChild();
      }
      return appender.getPosVar();
    }(appender);
    // FIXME: need to initialize posAccumulationVar
    Expr posAccumulationVar = ir::Var::make(pos.as<Var>()->name + "_acc", pos.as<Var>()->type, true);
    posAccumulationVars.push_back(posAccumulationVar);

    Expr beginPos = appender.getBeginVar();
    Expr parentPos = appender.getParent().getPosVar();

    Expr prevCoordIterator = (previousIteratorCoord.size() > 0) ? coordinateScanVarsMap[previousIteratorCoord.back()][3] : 0;
    Expr posNext = ir::Var::make(pos.as<Var>()->name + "_next", pos.as<Var>()->type);
    Stmt posNextDecl = ir::VarDecl::make(posNext,
                                         ir::Add::make(ir::RMW::make(posAccumulationVar, 0, reg, Expr(), SpatialRMWoperators::Add,
                                                       SpatialMemOrdering::Ordered),
                                                       ir::Div::make(prevCoordIterator, 100000)));
    result.push_back(posNextDecl);
    Stmt beginVarDecl = ir::VarDecl::make(appender.getBeginVar(), ir::Sub::make(posNext, reg));
    result.push_back(beginVarDecl);
    varMap[appender] = appender.getBeginVar();


    Stmt appendEdges = appender.getAppendEdges(parentPos, beginPos, posNext);

    result.push_back(appendEdges);
  }
  Stmt appendPosition = result.empty() ? Stmt() : Block::make(result);

  return Block::blanks(Block::make(regDecl, reduction), appendPosition);
}

Stmt LowererImplSpatial::generateIteratorBitVectors(IndexStmt statement, ir::Expr coordinate, IndexVar coordinateVar,
                                                    MergePoint point, map<Iterator, ir::Expr> &bvRawMap,
                                                    map<Iterator, ir::Expr> &bvMap) {

  taco_iassert(indexVartoBitVarMap.find(coordinateVar) != indexVartoBitVarMap.end()) << "Iterator not found in bit "
                                                                                        "variable map defined by global "
                                                                                        "metadata environment";
  auto bitLen = ir::Mul::make(ir::Literal::make(32), indexVartoBitVarMap.at(coordinateVar));

  vector<ir::Stmt> stmts;
  vector<ir::Stmt> deepFIFOCopyDecls;
  for (auto &iterator : point.iterators()) {
    auto crd = iterator.getMode().getModePack().getArray(1);
    auto parentPos = iterator.getParent().getPosVar();
    vector<ir::Expr> bounds = {iterator.getBeginVar(), iterator.getEndVar()};
    // auto bounds = iterator.posBounds(parentPos);

    taco_iassert(bvRawMap.find(iterator) != bvRawMap.end()) << "Iterator (" << iterator
                                                            << ") cannot be found in the variable map";
    auto fifo = bvRawMap[iterator];

    // FIXME: var isn't a pointer but can only allocate pointer type
    Expr var = ir::Var::make(iterator.getTensor().as<Var>()->name + "_bvRaw",
                             Datatype::UInt32, true, false, false,
                             MemoryLocation::SpatialFIFO);
    bvRawMap[iterator] = var;
    // FIXME: do not hardcode size of 16. Also fix this so that the memDecl is a size and not the RHS
    Stmt allocate = ir::Allocate::make(var, ir::Literal::make(16), false, false,
                                       false, MemoryLocation::SpatialFIFO);
    stmts.push_back(allocate);

    auto varDeep = ir::Var::make(iterator.getTensor().as<Var>()->name + "_bv",
                                 Datatype::UInt32, true, false, false,
                                 MemoryLocation::SpatialFIFO);
    bvMap.insert({iterator, varDeep});
    allocate = ir::Allocate::make(varDeep, ir::Literal::make(4096), false, false,
                                  false, MemoryLocation::SpatialFIFO);
    stmts.push_back(allocate);


    // FIXME: do not hardcode out_bitcnt
    auto genBitVector = ir::GenBitVector::make(ir::Literal::zero(var.type()), bitLen,
                                               iterator.getOccupancyVar(), fifo, var);

    stmts.push_back(genBitVector);


    auto fifoCopy = ir::Store::make(varDeep, ir::Literal::zero(Int64),
                                    ir::Load::make(var, ir::Literal::zero(Int()), MemoryLocation::SpatialFIFO),
                                    MemoryLocation::SpatialFIFO, MemoryLocation::SpatialFIFO);

    deepFIFOCopyDecls.push_back(fifoCopy);
  }

  Stmt deepFIFOCopyBody = Block::make(deepFIFOCopyDecls);

  taco_iassert(indexVartoBitVarMap.find(coordinateVar) != indexVartoBitVarMap.end())
    << "Cannot find indexVar in metadata "
       "environment map for the "
       "\"Tn_dimension_max_bits\" variable";
  auto stop = indexVartoBitVarMap[coordinateVar];
  Stmt deepFIFOCopyLoop = ir::For::make(ir::Var::make("bv_temp", Int64), ir::Literal::zero(Int64), stop,
                                        ir::Literal::make(1),
                                        funcEnvMap.at("ip"), deepFIFOCopyBody);

  return Block::blanks(Block::make(stmts), deepFIFOCopyLoop);
}

// FIXME: only do this if the iterator TensorVar is stored in DRAM base don tensorVar memoryLocation
Stmt LowererImplSpatial::loadDRAMtoFIFO(IndexStmt statement, MergePoint point, map<Iterator, ir::Expr> &varMap) {
  vector<ir::Stmt> stmts;
  for (auto &iterator : point.iterators()) {
    auto crd = iterator.getMode().getModePack().getArray(1);
    auto parentPos = iterator.getParent().getPosVar();
    vector<ir::Expr> bounds = {iterator.getBeginVar(), iterator.getOccupancyVar()};


    // FIXME: var isn't a pointer but can only allocate pointer type
    Expr var = ir::Var::make(iterator.getTensor().as<Var>()->name + "_fifo",
                             iterator.getTensor().as<Var>()->type, true, false, false,
                             MemoryLocation::SpatialFIFO);
    varMap[iterator] = var;

    // FIXME: do not hardcode size of 16. Also fix this so that the memDecl is a size and not the RHS
    Stmt allocate = ir::Allocate::make(var, ir::Literal::make(16), false, false,
                                       false, MemoryLocation::SpatialFIFORetimed);
    stmts.push_back(allocate);

    auto storeEnd = iterator.getOccupancyVar();
    auto data = ir::LoadBulk::make(crd, bounds[0], bounds[1]);
    Stmt store = ir::StoreBulk::make(var, data,
                                     MemoryLocation::SpatialFIFORetimed, MemoryLocation::SpatialDRAM);

    stmts.push_back(store);
  }
  return Block::make(stmts);
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

Stmt LowererImplSpatial::appendCoordinate(vector<Iterator> appenders, Expr coord) {
  vector<Stmt> result;
  for (auto &appender : appenders) {
    Expr pos = appender.getPosVar();
    Iterator appenderChild = appender.getChild();

    if (appenderChild.defined() && appenderChild.isBranchless()) {
      // Already emitted assembly code for current level when handling
      // branchless child level, so don't emit code again.
      continue;
    }

    vector<Stmt> appendStmts;

    if (coordinateScanVarsMap.count(coord)) {
      appendStmts.push_back(appender.getAppendCoord(pos, coordinateScanVarsMap.at(coord)[0]));
    } else {
      appendStmts.push_back(appender.getAppendCoord(pos, coord));
    }

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
        if (coordinateScanVarsMap.count(coord)) {
          appendStmts.push_back(appender.getAppendCoord(pos, coordinateScanVarsMap.at(coord)[0]));
        } else {
          appendStmts.push_back(appender.getAppendCoord(pos, coord));
        }
      }
    }

    if (generateAssembleCode() || isLastAppender(appender)) {
      //appendStmts.push_back(compoundAssign(pos, 1));

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

Stmt
LowererImplSpatial::codeToInitializeIteratorVar(Iterator iterator, vector<Iterator> iterators, vector<Iterator> rangers,
                                                vector<Iterator> mergers, Expr coordinate, IndexVar coordinateVar) {
  vector<Stmt> result;
  taco_iassert(iterator.hasPosIter() || iterator.hasCoordIter() ||
               iterator.isDimensionIterator());

  Expr iterVar = iterator.getIteratorVar();
  Expr endVar = iterator.getEndVar();
  Expr beginVar = iterator.getBeginVar();
  if (iterator.hasPosIter()) {
    Expr parentPos = iterator.getParent().getPosVar();
    if (iterator.getParent().isRoot() || iterator.getParent().isUnique()) {
      // E.g. a compressed mode without duplicates
      ModeFunction bounds = iterator.posBounds(parentPos);
      result.push_back(bounds.compute());
      // if has a coordinate ranger then need to binary search
      if (any(rangers,
              [](Iterator it) { return it.isDimensionIterator(); })) {

        Expr binarySearchTarget = provGraph.deriveCoordBounds(definedIndexVarsOrdered, underivedBounds,
                                                              indexVarToExprMap, this->iterators)[coordinateVar][0];
        if (binarySearchTarget != underivedBounds[coordinateVar][0]) {
          // If we have a window, then we need to project up the binary search target
          // into the window rather than the beginning of the level.
          if (iterator.isWindowed()) {
            binarySearchTarget = this->projectCanonicalSpaceToWindowedPosition(iterator, binarySearchTarget);
          }
          result.push_back(VarDecl::make(iterator.getBeginVar(), binarySearchTarget));

          vector<Expr> binarySearchArgs = {
            iterator.getMode().getModePack().getArray(1), // array
            bounds[0], // arrayStart
            bounds[1], // arrayEnd
            iterator.getBeginVar() // target
          };
          result.push_back(
            VarDecl::make(iterVar, Call::make("taco_binarySearchAfter", binarySearchArgs, iterVar.type())));
        } else {
          result.push_back(VarDecl::make(beginVar, bounds[0]));
        }
      } else {
        auto bound = bounds[0];
        // If we have a window on this iterator, then search for the start of
        // the window rather than starting at the beginning of the level.
        if (iterator.isWindowed()) {
          bound = this->searchForStartOfWindowPosition(iterator, bounds[0], bounds[1]);
        }
        result.push_back(VarDecl::make(beginVar, bound));
      }

      result.push_back(VarDecl::make(endVar, bounds[1]));
      result.push_back(VarDecl::make(iterator.getOccupancyVar(), ir::Sub::make(endVar, beginVar)));
    } else {
      taco_iassert(iterator.isOrdered() && iterator.getParent().isOrdered());
      taco_iassert(iterator.isCompact() && iterator.getParent().isCompact());

      // E.g. a compressed mode with duplicates. Apply iterator chaining
      Expr parentSegend = iterator.getParent().getSegendVar();
      ModeFunction startBounds = iterator.posBounds(parentPos);
      ModeFunction endBounds = iterator.posBounds(ir::Sub::make(parentSegend, 1));
      result.push_back(startBounds.compute());
      result.push_back(VarDecl::make(beginVar, startBounds[0]));
      result.push_back(endBounds.compute());
      result.push_back(VarDecl::make(endVar, endBounds[1]));
      result.push_back(VarDecl::make(iterator.getOccupancyVar(), ir::Sub::make(endVar, beginVar)));
    }
  } else if (iterator.hasCoordIter()) {
    // E.g. a hasmap mode
    vector<Expr> coords = coordinates(iterator);
    coords.erase(coords.begin());
    ModeFunction bounds = iterator.coordBounds(coords);
    result.push_back(bounds.compute());
    result.push_back(VarDecl::make(beginVar, bounds[0]));
    result.push_back(VarDecl::make(endVar, bounds[1]));
    result.push_back(VarDecl::make(iterator.getOccupancyVar(), ir::Sub::make(endVar, beginVar)));
  } else if (iterator.isDimensionIterator()) {
    // A dimension
    // If a merger then initialize to 0
    // If not then get first coord value like doing normal merge

    // If derived then need to recoverchild from this coord value
    bool isMerger = find(mergers.begin(), mergers.end(), iterator) != mergers.end();
    if (isMerger) {
      Expr coord = coordinates(vector<Iterator>({iterator}))[0];
      result.push_back(VarDecl::make(coord, 0));
    } else {
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
      Stmt end_decl = VarDecl::make(iterator.getEndVar(),
                                    provGraph.deriveIterBounds(iterator.getIndexVar(), definedIndexVarsOrdered,
                                                               underivedBounds, indexVarToExprMap, this->iterators)[1]);
      result.push_back(end_decl);
    }
  }
  return result.empty() ? Stmt() : Block::make(result);
}

// Need to pass in TensorVar list of all taco_tensor_t's to create A_nnz, B_nnz, etc.
Stmt LowererImplSpatial::generateGlobalEnvironmentVars(IndexStmt stmt) {
  envValMap = {{"ip",            16},
               {"sp",            16},
               {"bp",            1},
               {"nnz_max",       ir::Mul::make(128, ir::Mul::make(1024, 1024))},
               {"dimension_max", 65536},
               {"nnz_accel_max", 65536}};
  if (isa<SuchThat>(stmt)) {
    auto envs = stmt.as<SuchThat>().getEnvironment();
    for (auto &env : envs) {
      envValMap[env.getName()] = (int) env.getValue();
    }
  }

  vector<Stmt> envDecls;
  // FIXME: currently hardcoded
  for (auto it = envValMap.begin(); it != envValMap.end(); it++) {
    auto var = ir::Var::make(it->first, Int());
    auto varDecl = ir::VarDecl::make(var, it->second);
    envDecls.push_back(varDecl);
    funcEnvMap.insert({it->first, var});
  }

  vector<Stmt> maxVars;
  for (const IndexVar &indexVar : provGraph.getAllIndexVars()) {
    auto ivarMax = ir::Var::make(indexVar.getName() + "_max", Int(), true, false, false, MemoryLocation::SpatialArgIn);
    indexVartoMaxVar.insert({indexVar, ivarMax});
    auto allocate = ir::Allocate::make(ivarMax, 1, false, Expr(), false, MemoryLocation::SpatialArgIn);
    maxVars.push_back(allocate);
  }
  std::sort(maxVars.begin(), maxVars.end(),
            [&](const Stmt a,
                const Stmt b) -> bool {
              // first, use a total order of outputs,inputs
              auto aVarname = a.as<Allocate>()->var.as<Var>()->name;
              auto bVarname = b.as<Allocate>()->var.as<Var>()->name;
              return aVarname < bVarname;
            });

  return (Block::blanks(FuncEnv::make(Block::make(envDecls)), Block::make(maxVars)));
}

Stmt LowererImplSpatial::generateAccelEnvironmentVars() {

  vector<Stmt> bitVars;
  for (const IndexVar &indexVar : provGraph.getAllIndexVars()) {
    auto maxBitsVar = ir::Var::make(indexVar.getName() + "_max_bits", Int());
    indexVartoBitVarMap.insert({indexVar, maxBitsVar});

    taco_iassert(indexVartoMaxVar.find(indexVar) != indexVartoMaxVar.end())
      << "The index variable max bits var (i_max_bits) "
         "cannot be defined since the index variable max var "
         "(i_max) is not defined";
    auto eq = ir::Mul::make(16, ir::Div::make(ir::Add::make(indexVartoMaxVar.at(indexVar), 511), 512));
    auto maxBits = ir::VarDecl::make(maxBitsVar, eq);
    bitVars.push_back(maxBits);
  }


  return Block::blanks(Block::make(bitVars));
}

Stmt LowererImplSpatial::addAccelEnvironmentVars() {
  return codeToInitializePosAccumulators();
}

Stmt LowererImplSpatial::codeToInitializePosAccumulators() {
  taco_iassert(funcEnvMap.count("ip") > 0) << "Cannot find the inner-parallelization factor "
                                              "variable 'ip' in the function environment map";
  taco_iassert(funcEnvMap.count("bp") > 0) << "Cannot find the body-parallelization variable "
                                              "variable 'bp' in the function environment map";
  vector<Stmt> resultStmts;
  for (int i = 0; i < (int)posAccumulationVars.size(); i++) {
    auto posAccVar = posAccumulationVars.at(i);
    Stmt allocPosAccVar = ir::Allocate::make(posAccVar,funcEnvMap.at("ip"), false, Expr(), false,
                                          MemoryLocation::SpatialSparseParSRAM);
    resultStmts.push_back(allocPosAccVar);

    auto innerLoopIdxVar = ir::Var::make("j_temp", Int());
    Stmt innerLoopBody = ir::CallStmt::make(
      ir::RMW::make(posAccVar, innerLoopIdxVar, 0, Expr(), SpatialRMWoperators::Write, SpatialMemOrdering::Ordered));
    auto innerLoop = ir::For::make(innerLoopIdxVar, 0, funcEnvMap.at("ip"), 1, funcEnvMap.at("ip"), innerLoopBody);
    auto outerLoopIdxVar = ir::Var::make("i_temp", Int());
    auto outerLoop = ir::For::make(outerLoopIdxVar, 0, funcEnvMap.at("bp"), 1, funcEnvMap.at("bp"), innerLoop);
    resultStmts.push_back(outerLoop);
  }

  return Block::make(resultStmts);
}

Stmt LowererImplSpatial::generateAppendPositions(vector<Iterator> appenders) {
  // TODO: generate append position needs to add values to the counter

  vector<Stmt> result;
  for (Iterator appender : appenders) {
    if (appender.isBranchless() ||
        isAssembledByUngroupedInsertion(appender.getTensor())) {
      continue;
    }

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
  return result.empty() ? Stmt() : Block::make(result);
}

Stmt LowererImplSpatial::generateAppendPositionsForallPos(vector<Iterator> appenders, Expr pos, Expr coordinate) {
  vector<Stmt> result;
  for (Iterator appender : appenders) {
    if (appender.isBranchless() ||
        isAssembledByUngroupedInsertion(appender.getTensor())) {
      continue;
    }

    Expr beginPos = appender.getBeginVar();
    Expr parentPos = appender.getParent().getPosVar();

    if (isa<Var>(parentPos))
      result.push_back(ir::VarDecl::make(parentPos, coordinate));
    auto appendEdges = appender.getAppendEdges(parentPos, beginPos, pos);
    result.push_back(appendEdges);
  }
  return result.empty() ? Stmt() : Block::make(result);
}

std::vector<ir::Expr> LowererImplSpatial::getAllTemporaryModeArrays(Where where) {
  vector<Expr> arrays;

  Access temporaryAccess = getResultAccesses(where.getProducer()).first[0];
  auto temporaryIterators = getIterators(temporaryAccess);
  for (auto &iterator : temporaryIterators) {
    if (iterator.getMode().defined()) {
      if (iterator.getMode().getModeFormat() == dense) {
        arrays.push_back(iterator.getMode().getModePack().getArray(0));
      } else if (iterator.getMode().getModeFormat() == sparse) {
        arrays.push_back(iterator.getMode().getModePack().getArray(0));
        arrays.push_back(iterator.getMode().getModePack().getArray(1));
      }
    }
  }
  return arrays;
}

ir::Stmt LowererImplSpatial::generateOPMemLoads(Forall forall, Stmt &forallBodyStmt, Expr startBound,
                                                Expr endBound, TensorVar tns) {

  struct GetTensorProperties : IRVisitor {
    using IRVisitor::visit;
    TensorVar tv;
    map<TensorVar, Expr> tvs;
    vector<Expr> gps;

    GetTensorProperties(TensorVar tv, map<TensorVar, Expr> tvs) : tv(tv), tvs(tvs) {}

    void visit(const GetProperty *op) {
      op->tensor.accept(this);
      if (tvs.count(tv) > 0 && op->tensor == tvs.at(tv)) {
        if ((op->property == TensorProperty::Indices && op->index == 1) || op->property == TensorProperty::Values) {
          gps.push_back(GetProperty::make(op->tensor, op->property, op->mode, op->index, op->name));
        }
      }
    }

    vector<Expr> getProperties(Stmt stmt) {
      stmt.accept(this);
      return gps;
    }
  };

  struct TransformLocateDecl : IRRewriter {
    using IRRewriter::rewrite;
    Forall forall;
    TensorVar tv;

    bool varDecl = false;

    TransformLocateDecl(TensorVar tv, Forall forall) : tv(tv), forall(forall) {}

    void visit(const VarDecl *op) {
      if (isa<Var>(op->var)) {
        const Var *var = op->var.as<Var>();
        if (var->name == forall.getIndexVar().getName() + tv.getName()) {
          varDecl = true;
          auto rhs = rewrite(op->rhs);
          stmt = VarDecl::make(op->var, rhs, op->mem);
          varDecl = false;
          return;
        }
      }
      stmt = op;
    }

    void visit(const ir::Add *op) {
      if (varDecl) {
        expr = ir::Add::make(0, op->b);
        return;
      }
      expr = op;
    }
  };


  auto block = forallBodyStmt;
  if (std::find(forall.getCommunicateTensors().begin(), forall.getCommunicateTensors().end(), tns) !=
      forall.getCommunicateTensors().end()) {  // || previous ir::For has par != 1
    auto accessGPs = GetTensorProperties(tns, tensorVars).getProperties(forallBodyStmt);

    vector<Stmt> returnStmts;
    for (auto &expr : accessGPs) {
      auto gp = expr.as<GetProperty>();
      auto memoryLoc = tns.getMemoryLocation();

      Expr dim;
      switch (memoryLoc) {
        case MemoryLocation::SpatialFIFORetimed:
        case MemoryLocation::SpatialFIFO:
          dim = 16;
          break;
        default:
          dim = ir::Div::make(funcEnvMap["nnz_accel_max"], 4);
          break;
      }

      auto gpVar = ir::Var::make(gp->name, gp->type, true, false, false, memoryLoc);
      auto allocate = ir::Allocate::make(gp, dim, false, Expr(), false, memoryLoc);
      tensorPropertyVars[expr] = gpVar;

      auto dramVar = ir::Var::make(gp->name + "_dram", gp->type, true, false, false, MemoryLocation::SpatialDRAM);
      auto load = ir::LoadBulk::make(dramVar, startBound, endBound, funcEnvMap["ip"]);
      auto store = ir::StoreBulk::make(gp, load, memoryLoc, MemoryLocation::SpatialDRAM);

      returnStmts.push_back(allocate);
      returnStmts.push_back(store);

    }
    block = Block::make(returnStmts);
    block = addGPLoadFlag(Block::blanks(block, forallBodyStmt), tns, tensorVars);
    block = TransformLocateDecl(tns, forall).rewrite(block);
    block = TransformLocateDecl(tns, forall).rewrite(block);
  }
  return block;
}

ir::Stmt LowererImplSpatial::generateHoistedtensorAccess(Forall forall) {
  if (hoistedAccesses.count(forall.getIndexVar()) > 0 && !isa<Assignment>(forall.getStmt())) {
    vector<ir::Stmt> accessDecls;
    auto accesses = hoistedAccesses.at(forall.getIndexVar());
    for (auto& access : accesses) {

      auto tv = access.getTensorVar();
      auto tensor = tensorVars.at(tv);

      auto accessVar = ir::Var::make(tv.getName() + "_hoisted", tensor.type());
      auto accessExpr = lower(access);
      auto accessVarDecl = ir::VarDecl::make(accessVar, accessExpr);
      accessDecls.push_back(accessVarDecl);
      hoistedAccessVars[tv] = accessVar;
    }
    return Block::make(accessDecls);
  } else {
    return Block::make();
  }
}

ir::Stmt LowererImplSpatial::generateResultStore(Forall forall, vector<Iterator> appenders, Expr start, Expr end) {

  if (forall.getCommunicateTensors().size() > 0) {
    for (auto& tensor : forall.getCommunicateTensors()) {
      bool hasDimResult = false;
      match(forall,
            std::function<void(const WhereNode *, Matcher* ctx)>([&](const WhereNode *op, Matcher* ctx) {
              ctx->match(op->consumer);
            }),
            std::function<void(const ForallNode *, Matcher* ctx)>([&](const ForallNode *op, Matcher* ctx) {
             ctx->match(op->stmt);
            }),
            std::function<void(const AssignmentNode *, Matcher* ctx)>([&](const AssignmentNode *op, Matcher* ctx) {
              if (op->lhs.getTensorVar() == tensor)
                hasDimResult = true;
            }));

      if (hasResult(appenders, tensor) || hasDimResult) {
        hasResultCommunicate = true;
        Expr startBound = start;
        Expr EndBound = end;
        set_output_store(false);

        for (auto& access : resultTensorAccesses) {
          if (access.getTensorVar() == tensor) {
            bool seenForallIvar = false;
            for (auto& ivar : access.getIndexVars()) {
              if (ivar == forall.getIndexVar())
                seenForallIvar = true;
              else if (seenForallIvar) {
                startBound = ir::Mul::make(dimensions[ivar], startBound);
                EndBound = ir::Mul::make(dimensions[ivar], EndBound);
              }
            }
          }
        }


        auto load = ir::LoadBulk::make(getValuesArray(tensor), startBound, EndBound);

        auto dram = ir::Var::make(getValuesArray(tensor).as<GetProperty>()->name + "_dram",
                                  getValuesArray(tensor).type(), false, false, false, MemoryLocation::SpatialDRAM);
        auto decl = ir::StoreBulk::make(dram, load, MemoryLocation::SpatialDRAM, tensor.getMemoryLocation());
        return decl;

      }
    }
  }
  return Block::make();
}

bool LowererImplSpatial::hasResult(vector<Iterator> appenders, TensorVar tensor) {
  bool isResult = false;
  for (auto& appender : appenders) {
    if (appender.getTensor() == tensorVars[tensor]) {
      isResult = true;
    }
  }
  return isResult;
}

Stmt LowererImplSpatial::declLocatePosVars(vector<Iterator> locators) {
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
        auto coords = coordinates(locateIterator);
        auto expr = coords[coords.size() - 1];
        //Stmt declarePosVar = VarDecl::make(locateIterator.getPosVar(),
        //                                  expr);

        //result.push_back(declarePosVar);

        continue; // these will be recovered with separate procedure
      }
      do {
        auto coords = coordinates(locateIterator);
        // If this dimension iterator operates over a window, then it needs
        // to be projected up to the window's iteration space.
        if (locateIterator.isWindowed()) {
          auto expr = coords[coords.size() - 1];
          coords[coords.size() - 1] = this->projectCanonicalSpaceToWindowedPosition(locateIterator, expr);
        } else if (locateIterator.hasIndexSet()) {
          // If this dimension iterator operates over an index set, follow the
          // indirection by using the locator access the index set's crd array.
          // The resulting value is where we should locate into the actual tensor.
          auto expr = coords[coords.size() - 1];
          auto indexSetIterator = locateIterator.getIndexSetIterator();
          auto coordArray = indexSetIterator.posAccess(expr, coordinates(indexSetIterator)).getResults()[0];
          coords[coords.size() - 1] = coordArray;
        }
        ModeFunction locate = locateIterator.locate(coords);
        taco_iassert(isValue(locate.getResults()[1], true));
        Stmt declarePosVar = VarDecl::make(locateIterator.getPosVar(),
                                           locate.getResults()[0], locateIterator.getMode().getMemoryLocation());

//        if (locator.getMode().getMemoryLocation() != MemoryLocation::SpatialFIFO &&
//            locator.getMode().getMemoryLocation() != MemoryLocation::SpatialFIFORetimed)
          result.push_back(declarePosVar);

        if (locateIterator.isLeaf()) {
          break;
        }

        locateIterator = locateIterator.getChild();
      } while (locateIterator.hasLocate() &&
               accessibleIterators.contains(locateIterator));
    }
  }
  return result.empty() ? Stmt() : Block::make(result);
}

Stmt LowererImplSpatial::generateOPResultHoist(Forall forall, Stmt body) {
  // If outer par is > 1, move result allocation into forall
  vector<Stmt> allocateResultVals;
  if (forall == outerForall && envValMap.at("bp").as<ir::Literal>()->getTypedVal() > 1) {
    // Get all properties from result
    // move result allocation here.. but how?
    for (auto& access : resultTensorAccesses) {
      auto var = access.getTensorVar();
      auto vals = getValuesArray(var);
      auto memLoc = access.getTensorVar().getMemoryLocation();
      if (memLoc != MemoryLocation::SpatialDRAM &&
          memLoc != MemoryLocation::SpatialSparseDRAM &&
          memLoc != MemoryLocation::SpatialSparseDRAMFalse) {
        Expr size;
        switch (memLoc) {
          case MemoryLocation::SpatialFIFO:
          case MemoryLocation::SpatialFIFORetimed:
            size = 16;
            break;
          default:
            size = funcEnvMap.at("nnz_accel_max");
            break;
        }

        auto allocate = ir::Allocate::make(vals, size, false, Expr(), false, memLoc);
        allocate = addGPLoadFlag(allocate, access.getTensorVar(), tensorVars);
        allocateResultVals.push_back(allocate);

        body = addGPLoadFlag(body, access.getTensorVar(), tensorVars);

        if (util::contains(var.getFormat().getModeFormats(), sparse)) {
          for (int i = 0; i < (int) var.getFormat().getModeFormats().size(); i++) {
            auto modeFormat = var.getFormat().getModeFormats()[i];
            if (modeFormat == sparse || modeFormat == compressed) {
              auto crd = GetProperty::make(getTensorVar(var), TensorProperty::Indices, i, 1,
                                           var.getName() + to_string(i+1) + "_crd", false);
              auto crdAllocate = ir::Allocate::make(crd, size, false, Expr(), false, memLoc);
              crdAllocate = addGPLoadFlag(crdAllocate, access.getTensorVar(), tensorVars);
              allocateResultVals.push_back(crdAllocate);

              body = addGPLoadFlag(body, access.getTensorVar(), tensorVars);
            }
          }
        }
        if (var.getFormat().getModeFormats().size() > 0 && var.getFormat().getModeFormats().back() == sparse) {
          auto pos = GetProperty::make(getTensorVar(var), TensorProperty::Indices, var.getOrder() - 1, 0,
                                       var.getName() + to_string(var.getOrder()) + "_pos", false);
          auto posAllocate = ir::Allocate::make(pos, funcEnvMap.at("nnz_accel_max"), false, Expr(), false, MemoryLocation::SpatialSRAM);
          posAllocate = addGPLoadFlagAll(posAllocate, access.getTensorVar(), tensorVars);
          allocateResultVals.push_back(posAllocate);

          body = addGPLoadFlagAll(body, access.getTensorVar(), tensorVars);
        }
      }
    }
  }
  return Block::make(allocateResultVals);
}

} // namespace taco
