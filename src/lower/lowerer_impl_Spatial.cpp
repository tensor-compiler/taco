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
        return Block::make(hoistedAccesses, computeStmt);
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
          std::function<void(const AccessNode*)>([&](const AccessNode* op) {
            TensorVar var = Access(op).getTensorVar();
            if (var.getMemoryLocation() == MemoryLocation::SpatialSparseDRAM)
              hasSparseDRAMAccess = true;
          })
    );

    return hasSparseDRAMAccess;
  }

  Stmt LowererImplSpatial::hoistSparseDRAMAccesses(IndexExpr expression) {
    vector<Stmt> hoistedStmts;
    match(expression,
          std::function<void(const AccessNode*)>([&](const AccessNode* op) {
            TensorVar var = Access(op).getTensorVar();
            if (!isScalar(var.getType()) && var.getMemoryLocation() == MemoryLocation::SpatialSparseDRAM) {
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

    if (var.getMemoryLocation() == MemoryLocation::SpatialSparseDRAM && sparseDRAMAccessMap.find(var) != sparseDRAMAccessMap.end()) {
      return sparseDRAMAccessMap.at(var);
    }

    return getIterators(access).back().isUnique()
           ? Load::make(getValuesArray(var), generateValueLocExpr(access), var.getMemoryLocation())
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

    auto rootPoint = lattice.points()[0];
    // 1) Load from DRAM to FIFO
    map<Iterator, ir::Expr> varMap;
    Stmt memoryTransfer = loadDRAMtoFIFO(statement, rootPoint, varMap);

    // 2) generate BitVectors from FIFO to another FIFO
    map<Iterator, ir::Expr> bvMap;
    Stmt genBitVectors = generateIteratorBitVectors(statement, coordinate, coordinateVar, rootPoint, varMap, bvMap);

    // 3) Calculate pos arrays using scanner
    // An iteration lattice point represents a union when it contains more than 2 iterators and
    // dominates other points.
    bool isUnion = rootPoint.iterators().size() > 1 && lattice.subLattice(rootPoint).points().size() > 0;

    // Append position to the pos array
    Stmt appendPositions = generateIteratorAppendPositions(statement, coordinate, coordinateVar, rootPoint, appenders, varMap, isUnion);

    // 4) Calculate add
    Stmt computeLoop = generateIteratorComputeLoop(statement, coordinate, coordinateVar, rootPoint, lattice, bvMap,
                                                   reducedAccesses, isUnion);

    return Block::blanks(iteratorVarInits, memoryTransfer, genBitVectors, appendPositions, computeLoop);
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
    Expr coordinate = getCoordinateVar(forall.getIndexVar());
    Stmt declareCoordinate = Stmt();
    if (getProvGraph().isCoordVariable(forall.getIndexVar())) {
      Expr coordinateArray = iterator.posAccess(iterator.getPosVar(),
                                                coordinates(iterator)).getResults()[0];
      declareCoordinate = VarDecl::make(coordinate, coordinateArray, iterator.getMode().getMemoryLocation());

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
          return Block::make(boundsCompute, boundsDecl, regDecl, Reduce::make(loopVar, reg, startBound, endBound, 1,
                                                                              Block::make(declareCoordinate, Scope::make(reductionBody, reductionExpr)), true, forall.getNumChunks()));
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
          return Block::make(boundsCompute, boundsDecl, regDecl, Reduce::make(loopVar, reg, startBound, endBound, 1, Block::make(declareCoordinate, body), true,
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

  Stmt LowererImplSpatial::generateIteratorComputeLoop(IndexStmt statement, ir::Expr coordinate, IndexVar coordinateVar, MergePoint point,
                                                       MergeLattice lattice, map<Iterator, ir::Expr>& varMap, const std::set<Access>& reducedAccesses, bool isUnion) {

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

    Expr scan = Expr();
    Expr typeCase = Expr();
    if (point.iterators().size() == 1) {
      auto iterator = point.iterators()[0];
      // TODO: fix bitcnt and par factor later
      scan = ir::Scan::make(funcEnvMap.at("sp"),
                            bitLen,
                            varMap[iterator], Expr(), isUnion, false);


      typeCase = ir::TypeCase::make({uncompressedCrd, compressedCrd});
    } else {
      auto iterator1 = point.iterators()[0];
      auto iterator2 = point.iterators()[1];
      // TODO: fix bitcnt and par factor later
      scan = ir::Scan::make(funcEnvMap.at("sp"), bitLen, varMap[iterator1], varMap[iterator2], isUnion, false);

      typeCase = ir::TypeCase::make({uncompressedCrd, iterator1.getCoordVar(), compressedCrd, iterator2.getCoordVar()});
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

    for (auto& iterator : point.iterators()) {
      auto ternary = ir::Ternary::make(ir::Gte::make(iterator.getCoordVar(), ir::Literal::zero(Int())),
                                       ir::Add::make(iterator.getBeginVar(), iterator.getCoordVar()),
                                       ir::Literal::zero(Int()));
      Stmt iterDecl = ir::VarDecl::make(iterator.getIteratorVar(), ternary);
      iterVars.push_back(iterDecl);
    }

    Stmt body = lowerForallBody(coordinate, statement, {}, inserters,
                                appenders, reducedAccesses);

    body = Block::make(Block::make(iterVars), body);
    auto loop = ir::ForScan::make(typeCase, scan, body);

    return loop;
  }

  Stmt LowererImplSpatial::generateIteratorAppendPositions(IndexStmt statement, ir::Expr coordinate, IndexVar coordinateVar, MergePoint point,
                                                   std::vector<Iterator> appenders, map<Iterator, ir::Expr>& varMap, bool isUnion) {
    taco_iassert(point.iterators().size() <= 2) << "Spatial/Capstan hardware can only handle two sparse iterators"
                                                   "at once. To fix this please try inserting a temporary workspace";

    auto uncompressedCrd = ir::Var::make(coordinate.as<Var>()->name + "_uncomp", coordinate.as<Var>()->type);
    auto compressedCrd = ir::Var::make(coordinate.as<Var>()->name + "_comp", coordinate.as<Var>()->type);

    taco_iassert(indexVartoBitVarMap.find(coordinateVar) != indexVartoBitVarMap.end()) << "Iterator not found in bit "
                                                                                          "variable map defined by global "
                                                                                          "metadata environment";
    auto bitLen = ir::Mul::make(ir::Literal::make(32), indexVartoBitVarMap.at(coordinateVar));

    Expr scan = Expr();
    Expr typeCase = Expr();
    if (point.iterators().size() == 1) {
      auto iterator = point.iterators()[0];
      // TODO: fix bitcnt and par factor later
      scan = ir::Scan::make(1, bitLen,
                            varMap[iterator], Expr(), isUnion, true);


      typeCase = ir::TypeCase::make({uncompressedCrd, compressedCrd});
    } else {
      auto iterator1 = point.iterators()[0];
      auto iterator2 = point.iterators()[1];
      // TODO: fix bitcnt and par factor later
      scan = ir::Scan::make(1, bitLen, varMap[iterator1], varMap[iterator2], isUnion, true);

      typeCase = ir::TypeCase::make({uncompressedCrd, iterator1.getCoordVar(), compressedCrd, iterator2.getCoordVar()});
    }

    // Generate reduction register
    auto reg = ir::Var::make(coordinateVar.getName() + "_r_pos", coordinate.type(), false, false, false, MemoryLocation::SpatialReg);
    auto regDecl = ir::VarDecl::make(reg, ir::Literal::zero(reg.type()), MemoryLocation::SpatialReg);

    // Create reduction that counts up all of the bits in the bitvectors. This will be stored in the reulst Pos array
    // FIXME: weird spatial bug. See if uncompressedCrd is the correct variable here
    auto reductionBody = ir::Add::make(ir::Literal::make(1), ir::Div::make(compressedCrd, ir::Literal::make(10000)));
    auto reduction = ir::ReduceScan::make(typeCase, reg, scan, Stmt(), reductionBody);

    vector<Stmt> result;
    for (Iterator appender : appenders) {
//      if (appender.isBranchless() ||
//          isAssembledByUngroupedInsertion(appender.getTensor())) {
//        continue;
//      }

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

      Expr posNext = ir::Var::make(pos.as<Var>()->name + "_next", pos.as<Var>()->type);
      Stmt posNextDecl = ir::VarDecl::make(posNext, ir::RMW::make(posAccumulationVar, 0, reg, Expr(), SpatialRMWoperators::Add, SpatialMemOrdering::Ordered));
      result.push_back(posNextDecl);
      Stmt beginVarDecl = ir::VarDecl::make(appender.getBeginVar(), ir::Sub::make(posNext, reg));
      result.push_back(beginVarDecl);
      varMap[appender] = appender.getBeginVar();

      Expr beginPos = appender.getBeginVar();
      Expr parentPos = appender.getParent().getPosVar();
      Stmt appendEdges = appender.getAppendEdges(parentPos, beginPos, posNext);

      result.push_back(appendEdges);
    }
    Stmt appendPosition = result.empty() ? Stmt() : Block::make(result);

    return Block::blanks(Block::make(regDecl, reduction), appendPosition);
  }

  Stmt LowererImplSpatial::generateIteratorBitVectors(IndexStmt statement, ir::Expr coordinate, IndexVar coordinateVar,
                                                      MergePoint point, map<Iterator, ir::Expr>& bvRawMap, map<Iterator, ir::Expr>& bvMap) {

    taco_iassert(indexVartoBitVarMap.find(coordinateVar) != indexVartoBitVarMap.end()) << "Iterator not found in bit "
                                                                                          "variable map defined by global "
                                                                                          "metadata environment";
    auto bitLen = ir::Mul::make(ir::Literal::make(32), indexVartoBitVarMap.at(coordinateVar));

    vector<ir::Stmt> stmts;
    vector<ir::Stmt> deepFIFOCopyDecls;
    for (auto& iterator : point.iterators()) {
      auto crd = iterator.getMode().getModePack().getArray(1);
      auto parentPos = iterator.getParent().getPosVar();
      vector<ir::Expr> bounds = {iterator.getBeginVar(), iterator.getEndVar()};
      // auto bounds = iterator.posBounds(parentPos);

      taco_iassert(bvRawMap.find(iterator) != bvRawMap.end()) << "Iterator (" << iterator << ") cannot be found in the variable map";
      auto fifo = bvRawMap[iterator];

      // FIXME: var isn't a pointer but can only allocate pointer type
      Expr var = ir::Var::make(iterator.getTensor().as<Var>()->name + "_bvRaw",
                               iterator.getTensor().as<Var>()->type, true, false, false,
                               MemoryLocation::SpatialFIFO);
      bvRawMap[iterator] = var;
      // FIXME: do not hardcode size of 16. Also fix this so that the memDecl is a size and not the RHS
      Stmt allocate = ir::Allocate::make(var, ir::Literal::make(16), false, false,
                                         false, MemoryLocation::SpatialFIFO);
      stmts.push_back(allocate);

      auto varDeep = ir::Var::make(iterator.getTensor().as<Var>()->name + "_bv",
                                   iterator.getTensor().as<Var>()->type, true, false, false,
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

    taco_iassert(indexVartoBitVarMap.find(coordinateVar) != indexVartoBitVarMap.end()) << "Cannot find indexVar in metadata "
                                                                              "environment map for the "
                                                                              "\"Tn_dimension_max_bits\" variable";
    auto stop = indexVartoBitVarMap[coordinateVar];
    Stmt deepFIFOCopyLoop = ir::For::make(ir::Var::make("bv_temp", Int64), ir::Literal::zero(Int64), stop, ir::Literal::make(1), deepFIFOCopyBody);

    return  Block::blanks(Block::make(stmts), deepFIFOCopyLoop);
  }

  // FIXME: only do this if the iterator TensorVar is stored in DRAM base don tensorVar memoryLocation
  Stmt LowererImplSpatial::loadDRAMtoFIFO(IndexStmt statement, MergePoint point, map<Iterator, ir::Expr>& varMap) {
    vector<ir::Stmt> stmts;
    for (auto& iterator : point.iterators()) {
      auto crd = iterator.getMode().getModePack().getArray(1);
      auto parentPos = iterator.getParent().getPosVar();
      vector<ir::Expr> bounds = {iterator.getBeginVar(), iterator.getEndVar()};


      // FIXME: var isn't a pointer but can only allocate pointer type
      Expr var = ir::Var::make(iterator.getTensor().as<Var>()->name + "_fifo",
                                  iterator.getTensor().as<Var>()->type, true, false, false,
                                  MemoryLocation::SpatialFIFO);
      varMap[iterator] = var;

      // FIXME: do not hardcode size of 16. Also fix this so that the memDecl is a size and not the RHS
      Stmt allocate = ir::Allocate::make(var, ir::Literal::make(16), false, false,
                                         false, MemoryLocation::SpatialFIFO);
      stmts.push_back(allocate);

      auto storeEnd = iterator.getOccupancyVar();
      auto data = ir::LoadBulk::make(crd, bounds[0], bounds[1]);
      Stmt store = ir::StoreBulk::make(var, ir::Literal::zero(bounds[0].type()), storeEnd, data,
                                       MemoryLocation::SpatialFIFO, MemoryLocation::SpatialDRAM);

      stmts.push_back(store);
    }
    return  Block::make(stmts);
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
    for (auto& appender : appenders) {
      Expr pos = appender.getPosVar();
      Iterator appenderChild = appender.getChild();

      if (appenderChild.defined() && appenderChild.isBranchless()) {
        // Already emitted assembly code for current level when handling
        // branchless child level, so don't emit code again.
        continue;
      }

      vector<Stmt> appendStmts;

      //if (generateAssembleCode()) {
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

  Stmt LowererImplSpatial::codeToInitializeIteratorVar(Iterator iterator, vector<Iterator> iterators, vector<Iterator> rangers, vector<Iterator> mergers, Expr coordinate, IndexVar coordinateVar) {
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
                [](Iterator it){ return it.isDimensionIterator(); })) {

          Expr binarySearchTarget = provGraph.deriveCoordBounds(definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, this->iterators)[coordinateVar][0];
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
          }
          else {
            result.push_back(VarDecl::make(beginVar, bounds[0]));
          }
        }
        else {
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
    }
    else if (iterator.hasCoordIter()) {
      // E.g. a hasmap mode
      vector<Expr> coords = coordinates(iterator);
      coords.erase(coords.begin());
      ModeFunction bounds = iterator.coordBounds(coords);
      result.push_back(bounds.compute());
      result.push_back(VarDecl::make(beginVar, bounds[0]));
      result.push_back(VarDecl::make(endVar, bounds[1]));
      result.push_back(VarDecl::make(iterator.getOccupancyVar(), ir::Sub::make(endVar, beginVar)));
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
        Stmt end_decl = VarDecl::make(iterator.getEndVar(), provGraph.deriveIterBounds(iterator.getIndexVar(), definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, this->iterators)[1]);
        result.push_back(end_decl);
      }
    }
    return result.empty() ? Stmt() : Block::make(result);
  }

  // Need to pass in TensorVar list of all taco_tensor_t's to create A_nnz, B_nnz, etc.
  Stmt LowererImplSpatial::generateGlobalEnvironmentVars() {
    // FIXME: currently hardcoded
    auto innerParVar = ir::Var::make("ip", Int());
    auto innerPar = ir::VarDecl::make(innerParVar, 16);
    funcEnvMap.insert({"ip", innerParVar});

    auto scanParVar = ir::Var::make("sp", Int());
    auto scanPar = ir::VarDecl::make(scanParVar, 16);
    funcEnvMap.insert({"sp", scanParVar});

    auto bodyParVar = ir::Var::make("bp", Int());
    auto bodyPar = ir::VarDecl::make(bodyParVar, 1);
    funcEnvMap.insert({"bp", bodyParVar});

    auto maxNNZVar = ir::Var::make("nnz_max", Int());
    auto maxNNZVal = ir::Mul::make(128,ir::Mul::make(1024,1024));
    auto maxNNZ = ir::VarDecl::make(maxNNZVar, maxNNZVal);
    funcEnvMap.insert({"nnz_max", maxNNZVar});

    auto maxDimVar =  ir::Var::make("dimension_max", Int());
    auto maxDim = ir::VarDecl::make(maxDimVar, 65536);
    funcEnvMap.insert({"dimension_max", maxDimVar});

    auto maxNNZAccelVar =  ir::Var::make("nnz_accel_max", Int());
    auto maxNNZAccel = ir::VarDecl::make(maxNNZAccelVar, 65536);
    funcEnvMap.insert({"nnz_accel_max", maxNNZAccelVar});

    vector<Stmt> maxVars;
    for (const IndexVar& indexVar : provGraph.getAllIndexVars()) {
      auto ivarMax = ir::Var::make(indexVar.getName() + "_max", Int(), true, false, false, MemoryLocation::SpatialArgIn);
      indexVartoMaxVar.insert({indexVar, ivarMax});
      auto allocate = ir::Allocate::make(ivarMax, 1, false, Expr(), false, MemoryLocation::SpatialArgIn);
      maxVars.push_back(allocate);
    }
    return (Block::blanks(FuncEnv::make(Block::make(innerPar, scanPar, bodyPar, maxNNZ, maxNNZAccel, maxDim)), Block::make(maxVars)));
  }

  Stmt LowererImplSpatial::generateAccelEnvironmentVars() {

    vector<Stmt> bitVars;
    for (const IndexVar& indexVar : provGraph.getAllIndexVars()) {
      auto maxBitsVar = ir::Var::make(indexVar.getName() + "_max_bits", Int());
      indexVartoBitVarMap.insert({indexVar, maxBitsVar});

      taco_iassert(indexVartoMaxVar.find(indexVar) != indexVartoMaxVar.end()) << "The index variable max bits var (i_max_bits) "
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
    for (auto& posAccVar : posAccumulationVars) {
      auto allocPosAccVar = ir::Allocate::make(posAccVar, funcEnvMap.at("ip"), false, Expr(), false, MemoryLocation::SpatialSparseSRAM);
      resultStmts.push_back(allocPosAccVar);

      auto innerLoopIdxVar = ir::Var::make("j_temp", Int());
      Stmt innerLoopBody = ir::CallStmt::make(ir::RMW::make(posAccVar, innerLoopIdxVar, 0, Expr(), SpatialRMWoperators::Write, SpatialMemOrdering::Ordered));
      auto innerLoop = ir::For::make(innerLoopIdxVar, 0, funcEnvMap.at("bp"), 1, funcEnvMap.at("bp"), innerLoopBody);
      auto outerLoopIdxVar = ir::Var::make("i_temp", Int());
      auto outerLoop = ir::For::make(outerLoopIdxVar, 0, funcEnvMap.at("ip"), 1, funcEnvMap.at("ip"),  innerLoop);
      resultStmts.push_back(outerLoop);
    }

    return Block::make(resultStmts);
  }
} // namespace taco
