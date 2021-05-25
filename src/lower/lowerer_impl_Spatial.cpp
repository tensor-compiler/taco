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

using namespace std;
using namespace taco::ir;
using taco::util::combine;

namespace taco {

class LowererImplSpatial::Visitor : public IndexNotationVisitorStrict {
  public:
    Visitor(LowererImplSpatial* impl) : impl(impl) {}
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
    LowererImplSpatial* impl;
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
    void visit(const AssembleNode* node)      { stmt = impl->lowerAssemble(node); }
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

  LowererImplSpatial::LowererImplSpatial() : visitor(new Visitor(this)) {
  }

  Stmt LowererImplSpatial::lowerAssignment(Assignment assignment)
  {
    TensorVar result = assignment.getLhs().getTensorVar();

    if (generateComputeCode()) {
      Expr var = getTensorVar(result);
      Expr rhs = lower(assignment.getRhs());

      // Assignment to scalar variables.
      if (isScalar(result.getType())) {
        if (!assignment.getOperator().defined()) {
          return Assign::make(var, rhs, false, getAtomicParallelUnit());
          // TODO: we don't need to mark all assigns/stores just when scattering/reducing
        }
        else {
          taco_iassert(isa<taco::Add>(assignment.getOperator()));
          return compoundAssign(var, rhs, false, getAtomicParallelUnit());
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
            computeStmt = Store::make(values, loc, rhs, result.getMemoryLocation(), MemoryLocation::SpatialReg, false, getAtomicParallelUnit());
          }
        }
        else {
          computeStmt = compoundStore(values, loc, rhs, result.getMemoryLocation(), MemoryLocation::SpatialReg,false, getAtomicParallelUnit());
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

  Expr LowererImplSpatial::lowerAccess(Access access) {
    TensorVar var = access.getTensorVar();

    if (isScalar(var.getType())) {
      return getTensorVar(var);
    }

    return getIterators(access).back().isUnique()
           ? Load::make(getValuesArray(var), generateValueLocExpr(access))
           : getReducedValueVar(access);
  }

  ir::Expr LowererImplSpatial::getValuesArray(TensorVar var) const
  {
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
        Expr values = ir::Var::make(temporary.getName(),
                                    temporary.getType().getDataType(),
                                    true, false);
        taco_iassert(temporary.getType().getOrder() == 1) << " Temporary order was "
                                                          << temporary.getType().getOrder();  // TODO
        Dimension temporarySize = temporary.getType().getShape().getDimension(0);
        Expr size;
        if (temporarySize.isFixed()) {
          size = ir::Literal::make(temporarySize.getSize());
        } else if (temporarySize.isIndexVarSized()) {
          IndexVar var = temporarySize.getIndexVarSize();
          vector<Expr> bounds = getProvGraph().deriveIterBounds(var, getDefinedIndexVarsOrdered(), getUnderivedBounds(),
                                                           getIndexVarToExprMap(), getIterators());
          size = ir::Sub::make(bounds[1], bounds[0]);
        } else {
          taco_ierror; // TODO
        }

        // no decl needed for Spatial memory
//        Stmt decl = Stmt();
//        if ((isa<Forall>(where.getProducer()) && inParallelLoopDepth == 0) || !should_use_CUDA_codegen()) {
//          decl = (values, ir::Literal::make(0));
//        }
        Stmt allocate = Allocate::make(values, size);

        Expr p = Var::make("p" + temporary.getName(), Int());
        Stmt zeroInit = Store::make(values, p, ir::Literal::zero(temporary.getType().getDataType()));
        Stmt zeroInitLoop = For::make(p, 0, size, 1, zeroInit, LoopKind::Serial);

        /// Make a struct object that lowerAssignment and lowerAccess can read
        /// temporary value arrays from.
        TemporaryArrays arrays;
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

  Stmt LowererImplSpatial::lowerMergeLattice(MergeLattice lattice, IndexVar coordinateVar,
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

    return Block::make(iteratorVarInits, mergeLoops, appendPositions);
  }

  Stmt LowererImplSpatial::lowerMergePoint(MergeLattice pointLattice,
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

    cout << "DEBUG: Lower Merge Point"<< endl;
    cout << loadPosIterCoordinates << endl<< endl;
    cout << resolvedCoordinate << endl<< endl;
    cout << loadLocatorPosVars << endl<< endl;
    cout << deduplicationLoops << endl << endl;
    cout << caseStmts << endl << endl;
    cout << incIteratorVarStmts << endl<< endl;
    /// While loop over rangers
    return While::make(checkThatNoneAreExhausted(rangers),
                       Block::make(loadPosIterCoordinates,
                                   resolvedCoordinate,
                                   loadLocatorPosVars,
                                   deduplicationLoops,
                                   caseStmts,
                                   incIteratorVarStmts));
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

  Stmt LowererImplSpatial::lowerMergeCases(ir::Expr coordinate, IndexVar coordinateVar, IndexStmt stmt,
                                    MergeLattice lattice,
                                    const std::set<Access>& reducedAccesses)
  {
    vector<Stmt> result;

    vector<Iterator> appenders;
    vector<Iterator> inserters;
    tie(appenders, inserters) = splitAppenderAndInserters(lattice.results());

    // Just one iterator so no conditionals
    if (lattice.iterators().size() == 1) {
      cout << stmt << endl;
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
          if (!(getProvGraph().isCoordVariable(iterator.getIndexVar()) && getProvGraph().isDerivedFrom(iterator.getIndexVar(), coordinateVar))) {
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
        cases.push_back({taco::ir::conjunction(coordComparisons), body});
      }
      result.push_back(Case::make(cases, lattice.exact()));
    }

    return Block::make(result);
  }

Stmt LowererImplSpatial::lowerForallDimension(Forall forall,
                                               vector<Iterator> locators,
                                               vector<Iterator> inserters,
                                               vector<Iterator> appenders,
                                               set<Access> reducedAccesses,
                                               ir::Stmt recoveryStmt)
{
  Expr coordinate = getCoordinateVar(forall.getIndexVar());

  if (forall.getParallelUnit() != ParallelUnit::NotParallel && forall.getOutputRaceStrategy() == OutputRaceStrategy::Atomics) {
    markAssignsAtomicDepth++;
    atomicParallelUnit = forall.getParallelUnit();
  }

  Stmt body = lowerForallBody(coordinate, forall.getStmt(),
                              locators, inserters, appenders, reducedAccesses);


  if (forall.getParallelUnit() != ParallelUnit::NotParallel && forall.getOutputRaceStrategy() == OutputRaceStrategy::Atomics) {
    markAssignsAtomicDepth--;
  }

  body = Block::make({recoveryStmt, body});

  Stmt posAppend = generateAppendPositions(appenders);

  // Emit loop with preamble and postamble
  std::vector<ir::Expr> bounds = provGraph.deriveIterBounds(forall.getIndexVar(), definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);

  LoopKind kind = LoopKind::Serial;
  if (forall.getParallelUnit() == ParallelUnit::CPUVector && !ignoreVectorize) {
    kind = LoopKind::Vectorized;
  }
  else if (forall.getParallelUnit() != ParallelUnit::NotParallel
           && forall.getOutputRaceStrategy() != OutputRaceStrategy::ParallelReduction && !ignoreVectorize) {
    kind = LoopKind::Runtime;
  }

  if (forall.getParallelUnit() == ParallelUnit::Spatial && forall.getOutputRaceStrategy() == OutputRaceStrategy::SpatialReduction) {
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

      Stmt reductionBody = lowerForallReductionBody(coordinate, forall.getStmt(),
                                                    locators, inserters, appenders, reducedAccesses);
      reductionBody = Block::make({recoveryStmt, reductionBody});

      Expr reductionExpr = lower(forallExpr.getRhs());

      // FIXME: reduction can only handle adds for now
      taco_iassert(isa<taco::Add>(forallExpr.getOperator()));

      if (forallExpr.getOperator().defined()) {
        return Block::make(regDecl, Reduce::make(coordinate, reg, bounds[0], bounds[1], 1, Scope::make(reductionBody, reductionExpr), true, forall.getNumChunks()));
      }
    }
    if (forallReductions.find(forall) != forallReductions.end() && !provGraph.getParents(forall.getIndexVar()).empty()) {
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
        return Block::make(regDecl, Reduce::make(coordinate, reg, bounds[0], bounds[1], 1, body, true,
                                                 forall.getNumChunks()));
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

    Expr data = LoadBulk::make(valuesRhs, ir::Add::make(get<1>(locs[1]), bounds[0]), ir::Add::make(get<1>(locs[1]), bounds[1]));

    return Block::make(get<0>(locs[0]), StoreBulk::make(valuesLhs,
                                                        ir::Add::make(get<1>(locs[0]), bounds[0]),
                                                        ir::Add::make(get<1>(locs[0]), bounds[1]), data,
                                                        tensorLhs.getMemoryLocation(), tensorRhs.getMemoryLocation()));
  }

  auto returnExpr = Block::blanks(For::make(coordinate, bounds[0], bounds[1], 1, body,
                                            kind,
                                            ignoreVectorize ? ParallelUnit::NotParallel : forall.getParallelUnit(),
                                            ignoreVectorize ? 0 : forall.getUnrollFactor(), 0, forall.getNumChunks()),
                                  posAppend);

  return returnExpr;
}

  Stmt LowererImplSpatial::lowerForallPosition(Forall forall, Iterator iterator,
                                        vector<Iterator> locators,
                                        vector<Iterator> inserters,
                                        vector<Iterator> appenders,
                                        set<Access> reducedAccesses,
                                        ir::Stmt recoveryStmt)
  {
    cout << "Lower Forall Position" << endl;
    Expr coordinate = getCoordinateVar(forall.getIndexVar());
    Stmt declareCoordinate = Stmt();
    if (getProvGraph().isCoordVariable(forall.getIndexVar())) {
      Expr coordinateArray = iterator.posAccess(iterator.getPosVar(),
                                                coordinates(iterator)).getResults()[0];
      declareCoordinate = VarDecl::make(coordinate, coordinateArray);
    }
    if (forall.getParallelUnit() != ParallelUnit::NotParallel && forall.getOutputRaceStrategy() == OutputRaceStrategy::Atomics) {
      markAssignsAtomicDepth++;
    }

    Stmt body = lowerForallBody(coordinate, forall.getStmt(),
                                locators, inserters, appenders, reducedAccesses);

    if (forall.getParallelUnit() != ParallelUnit::NotParallel && forall.getOutputRaceStrategy() == OutputRaceStrategy::Atomics) {
      markAssignsAtomicDepth--;
    }

    body = Block::make(recoveryStmt, body);

    // Code to append positions
    Stmt posAppend = generateAppendPositions(appenders);

    // Code to compute iteration bounds
    Stmt boundsCompute;
    Expr startBound = 0;
    Expr endBound = iterator.getParent().getEndVar();
    Expr parentPos = iterator.getParent().getPosVar();
    if (!getProvGraph().isUnderived(iterator.getIndexVar())) {
      vector<Expr> bounds = getProvGraph().deriveIterBounds(iterator.getIndexVar(), getDefinedIndexVarsOrdered(), getUnderivedBounds(), getIndexVarToExprMap(), getIterators());
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

    auto loopVar = to<Var>(iterator.getPosVar());
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

    } else if (!isa<Var>(startBound)) {
      //TODO: Remove duplicate code here
      auto startVar = Var::make(loopVar->name + "_start", startBound.type());
      auto startBoundDecl = VarDecl::make(startVar, startBound);
      boundsDecl = startBoundDecl;
      startBound = startVar;
    } else if (!isa<Var>(endBound)) {
      //TODO: Remove duplicate code here
      auto endVar = Var::make(loopVar->name + "_end", endBound.type());
      auto endBoundDecl = VarDecl::make(endVar, endBound);
      boundsDecl = Block::make(boundsDecl, endBoundDecl);
      endBound = endVar;
    }

    LoopKind kind = LoopKind::Serial;
    if (forall.getParallelUnit() == ParallelUnit::CPUVector && !ignoreVectorize) {
      kind = LoopKind::Vectorized;
    }
    else if (forall.getParallelUnit() != ParallelUnit::NotParallel
             && forall.getOutputRaceStrategy() != OutputRaceStrategy::ParallelReduction && !ignoreVectorize) {
      kind = LoopKind::Runtime;
    }

    if (forall.getParallelUnit() == ParallelUnit::Spatial && forall.getOutputRaceStrategy() == OutputRaceStrategy::SpatialReduction) {
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

        Stmt reductionBody = lowerForallReductionBody(coordinate, forall.getStmt(),
                                                      locators, inserters, appenders, reducedAccesses);
        reductionBody = Block::make({recoveryStmt, reductionBody});

        Expr reductionExpr = lower(forallExpr.getRhs());

        taco_iassert(isa<taco::Add>(forallExpr.getOperator())) << "Spatial reductions can only handle additions";

        if (forallExpr.getOperator().defined()) {
          return Block::make(regDecl, Reduce::make(coordinate, reg, startBound, endBound, 1,
                                                   Scope::make(reductionBody, reductionExpr), true, forall.getNumChunks()));
        }
      }
      if (forallReductions.find(forall) != forallReductions.end() && !provGraph.getParents(forall.getIndexVar()).empty()) {
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
        auto parentVar = provGraph.getParents(forall.getIndexVar())[0];
        vector<IndexVar> children = provGraph.getChildren(parentVar);

        if (forallExpr.getOperator().defined()) {
          return Block::make(regDecl, Reduce::make(coordinate, reg, startBound, endBound, 1, body, true,
                                                   forall.getNumChunks()));
        }
      }
    }

    // Loop with preamble and postamble
    return Block::blanks(boundsCompute, boundsDecl,
                         For::make(loopVar, startBound, endBound, 1,
                                   Block::make(declareCoordinate, body),
                                   kind,
                                   ignoreVectorize ? ParallelUnit::NotParallel : forall.getParallelUnit(), ignoreVectorize ? 0 : forall.getUnrollFactor()),
                         posAppend);

  }

} // namespace taco
