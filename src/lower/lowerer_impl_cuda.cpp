//
// Created by 张 on 2022/3/10.
//

#include "taco/lower/lowerer_impl_cuda.h"
#include <taco/lower/mode_format_compressed.h>
#include "taco/lower/lowerer_impl_imperative.h"


#include "taco/index_notation/index_notation.h"

#include "taco/index_notation/index_notation_nodes.h"
#include "taco/index_notation/index_notation_visitor.h"
#include "taco/index_notation/index_notation_rewriter.h"

#include "taco/ir/ir.h"
#include "taco/ir/ir_generators.h"
#include "taco/ir/ir_visitor.h"
#include "taco/ir/simplify.h"
#include "taco/lower/iterator.h"
#include "taco/lower/merge_lattice.h"
#include "mode_access.h"
#include "taco/util/collections.h"

#include "taco/ir/workspace_rewriter.h"

using namespace std;
using namespace taco::ir;

namespace taco {
    class LowererImplCUDA::Visitor : public IndexNotationVisitorStrict {
    public:
        Visitor(LowererImplCUDA* impl):impl(impl) {}
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
        LowererImplCUDA* impl;
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
        void visit(const CallNode* node)      { expr = impl->lowerTensorOp(node); }
        void visit(const ReductionNode* node)  {
            taco_ierror << "Reduction nodes not supported in concrete index notation";
        }
        void visit(const IndexVarNode* node)       { expr = impl->lowerIndexVar(node); }
    };
    LowererImplCUDA::LowererImplCUDA(): visitor(new Visitor(this)) {}

    static bool returnsTrue(IndexExpr expr) {
        struct ReturnsTrue : public IndexExprRewriterStrict {
            void visit(const AccessNode* op) {
                if (op->isAccessingStructure) {
                    expr = op;
                }
            }

            void visit(const LiteralNode* op) {
                if (op->getDataType() == Bool && op->getVal<bool>()) {
                    expr = op;
                }
            }

            void visit(const NegNode* op) {
                expr = rewrite(op->a);
            }

            void visit(const AddNode* op) {
                if (rewrite(op->a).defined() || rewrite(op->b).defined()) {
                    expr = op;
                }
            }

            void visit(const MulNode* op) {
                if (rewrite(op->a).defined() && rewrite(op->b).defined()) {
                    expr = op;
                }
            }

            void visit(const CastNode* op) {
                expr = rewrite(op->a);
            }

            void visit(const CallNode* op) {
                const auto annihilator = findProperty<Annihilator>(op->properties);

                if (!annihilator.defined() || !annihilator.positions().empty()) {
                    return;
                }

                if (equals(annihilator.annihilator(), Literal(false))) {
                    for (const auto& arg : op->args) {
                        if (!rewrite(arg).defined()) {
                            return;
                        }
                    }
                    expr = op;
                } else {
                    for (const auto& arg : op->args) {
                        if (rewrite(arg).defined()) {
                            expr = op;
                            return;
                        }
                    }
                }
            }

            void visit(const SqrtNode* op) {}
            void visit(const SubNode* op) {}
            void visit(const DivNode* op) {}
            void visit(const CallIntrinsicNode* op) {}
            void visit(const ReductionNode* op) {}
            void visit(const IndexVarNode* op) {}
        };
        return ReturnsTrue().rewrite(expr).defined();
    }

    static bool needComputeValues(IndexStmt stmt, TensorVar tensor) {
        if (tensor.getType().getDataType() != Bool) {
            return true;
        }

        bool needComputeValue = false;
        match(stmt,
              function<void(const AssignmentNode*, Matcher*)>([&](
                      const AssignmentNode* n, Matcher* m) {
                  if (n->lhs.getTensorVar() == tensor && !returnsTrue(n->rhs)) {
                      needComputeValue = true;
                  }
              })
        );

        return needComputeValue;
    }
    vector<Stmt> LowererImplCUDA::codeToInitializeDenseAcceleratorArrays(Where where, bool parallel) {
        // if parallel == true, need to initialize dense accelerator arrays as size*numThreads
        // and rename all dense accelerator arrays to name + '_all'

        TensorVar temporary = where.getTemporary();

        // TODO: emit as uint64 and manually emit bit pack code
        const Datatype bitGuardType = taco::Bool;
        std::string bitGuardSuffix;
        if (parallel)
            bitGuardSuffix = "_already_set_all";
        else
            bitGuardSuffix = "_already_set";
        const std::string bitGuardName = temporary.getName() + bitGuardSuffix;

        Expr bitGuardSize = getTemporarySize(where);
        Expr maxThreads = ir::Call::make("omp_get_max_threads", {}, bitGuardSize.type());
        if (parallel)
            bitGuardSize = ir::Mul::make(bitGuardSize, maxThreads);

        const Expr alreadySetArr = ir::Var::make(bitGuardName,
                                                 bitGuardType,
                                                 true, false);

        // TODO: TACO should probably keep state on if it can use int32 or if it should switch to
        //       using int64 for indices. This assumption is made in other places of taco.
        const Datatype indexListType = taco::Int32;
        std::string indexListSuffix;
        if (parallel)
            indexListSuffix = "_index_list_all";
        else
            indexListSuffix = "_index_list";

        const std::string indexListName = temporary.getName() + indexListSuffix;
        const Expr indexListArr = ir::Var::make(indexListName,
                                                indexListType,
                                                true, false);

        // no decl for shared memory
        Stmt alreadySetDecl = Stmt();
        Stmt indexListDecl = Stmt();
        Stmt freeTemps = Block::make(Free::make(indexListArr), Free::make(alreadySetArr));
        if ((isa<Forall>(where.getProducer()) && inParallelLoopDepth == 0)) {
            alreadySetDecl = VarDecl::make(alreadySetArr, ir::Literal::make(0));
            indexListDecl = VarDecl::make(indexListArr, ir::Literal::make(0));
        }

        if (parallel) {
            whereToIndexListAll[where] = indexListArr;
            whereToBitGuardAll[where] = alreadySetArr;
        } else {
            const Expr indexListSizeExpr = ir::Var::make(indexListName + "_size", taco::Int32, false, false);
            tempToIndexList[temporary] = indexListArr;
            tempToIndexListSize[temporary] = indexListSizeExpr;
            tempToBitGuard[temporary] = alreadySetArr;
        }

        Stmt allocateIndexList = Allocate::make(indexListArr, bitGuardSize);
        Stmt allocateAlreadySet = Allocate::make(alreadySetArr, bitGuardSize);
        Expr p = Var::make("p" + temporary.getName(), Int());
        Stmt guardZeroInit = Store::make(alreadySetArr, p, ir::Literal::zero(bitGuardType));

        Stmt zeroInitLoop = For::make(p, 0, bitGuardSize, 1, guardZeroInit, LoopKind::Serial);
        Stmt inits = Block::make(alreadySetDecl, indexListDecl, allocateAlreadySet, allocateIndexList, zeroInitLoop);
        return {inits, freeTemps};


    }


// Returns true if the following conditions are met:
// 1) The temporary is a dense vector
// 2) There is only one value on the right hand side of the consumer
//    -- We would need to handle sparse acceleration in the merge lattices for
//       multiple operands on the RHS
// 3) The left hand side of the where consumer is sparse, if the consumer is an
//    assignment
// 4) CPU Code is being generated (TEMPORARY - This should be removed)
//    -- The sorting calls and calloc call in lower where are CPU specific. We
//       could map calloc to a cudaMalloc and use a library like CUB to emit
//       the sort. CUB support is built into CUDA 11 but not prior versions of
//       CUDA so in that case, we'd probably need to include the CUB headers in
//       the generated code.
    std::pair<bool,bool> LowererImplCUDA::canAccelerateDenseTemp(Where where) {
            return std::make_pair(false, false);
    }

// Code to initialize the local temporary workspace from the shared workspace
// in codeToInitializeTemporaryParallel for a SINGLE parallel unit
// (e.g.) the local workspace that each thread uses
    vector<Stmt> LowererImplCUDA::codeToInitializeLocalTemporaryParallel(Where where, ParallelUnit parallelUnit) {
        TensorVar temporary = where.getTemporary();
        vector<Stmt> decls;

        Expr tempSize = getTemporarySize(where);
        Expr threadNum = ir::Call::make("omp_get_thread_num", {}, tempSize.type());
        tempSize = ir::Mul::make(tempSize, threadNum);

        bool accelerateDense = canAccelerateDenseTemp(where).first;

        Expr values;
        if (util::contains(needCompute, temporary) &&
            needComputeValues(where, temporary)) {
            // Declare local temporary workspace array
            values = ir::Var::make(temporary.getName(),
                                   temporary.getType().getDataType(),
                                   true, false);
            Expr values_all = this->temporaryArrays[this->whereToTemporaryVar[where]].values;
            Expr tempRhs = ir::Add::make(values_all, tempSize);
            Stmt tempDecl = ir::VarDecl::make(values, tempRhs);
            decls.push_back(tempDecl);
        }
        /// Make a struct object that lowerAssignment and lowerAccess can read
        /// temporary value arrays from.
        TemporaryArrays arrays;
        arrays.values = values;
        this->temporaryArrays.insert({temporary, arrays});

        if (accelerateDense) {
            // Declare local index list array
            // TODO: TACO should probably keep state on if it can use int32 or if it should switch to
            //       using int64 for indices. This assumption is made in other places of taco.
            const Datatype indexListType = taco::Int32;
            const std::string indexListName = temporary.getName() + "_index_list";
            const Expr indexListArr = ir::Var::make(indexListName,
                                                    indexListType,
                                                    true, false);

            Expr indexList_all = this->whereToIndexListAll[where];
            Expr indexListRhs = ir::Add::make(indexList_all, tempSize);
            Stmt indexListDecl = ir::VarDecl::make(indexListArr, indexListRhs);
            decls.push_back(indexListDecl);

            // Declare local indexList size variable
            const Expr indexListSizeExpr = ir::Var::make(indexListName + "_size", taco::Int32, false, false);

            // Declare local already set array (bit guard)
            // TODO: emit as uint64 and manually emit bit pack code
            const Datatype bitGuardType = taco::Bool;
            const std::string bitGuardName = temporary.getName() + "_already_set";
            const Expr alreadySetArr = ir::Var::make(bitGuardName,
                                                     bitGuardType,
                                                     true, false);
            Expr bitGuard_all = this->whereToBitGuardAll[where];
            Expr bitGuardRhs = ir::Add::make(bitGuard_all, tempSize);
            Stmt bitGuardDecl = ir::VarDecl::make(alreadySetArr, bitGuardRhs);
            decls.push_back(bitGuardDecl);

            tempToIndexList[temporary] = indexListArr;
            tempToIndexListSize[temporary] = indexListSizeExpr;
            tempToBitGuard[temporary] = alreadySetArr;
        }
        return decls;
    }

// Code to initialize a temporary workspace that is SHARED across ALL parallel units.
// New temporaries are denoted by temporary.getName() + '_all'
// Currently only supports CPUThreads
    vector<Stmt> LowererImplCUDA::codeToInitializeTemporaryParallel(Where where, ParallelUnit parallelUnit) {
        TensorVar temporary = where.getTemporary();
        // For the parallel case, need to hoist up a workspace shared by all threads
        TensorVar temporaryAll = TensorVar(temporary.getName() + "_all", temporary.getType(), temporary.getFormat());
        this->whereToTemporaryVar[where] = temporaryAll;

        bool accelerateDense = canAccelerateDenseTemp(where).first;

        Stmt freeTemporary = Stmt();
        Stmt initializeTemporary = Stmt();

        // When emitting code to accelerate dense workspaces with sparse iteration, we need the following arrays
        // to construct the result indices
        if(accelerateDense) {
            vector<Stmt> initAndFree = codeToInitializeDenseAcceleratorArrays(where, true);
            initializeTemporary = initAndFree[0];
            freeTemporary = initAndFree[1];
        }

        Expr values;
        if (util::contains(needCompute, temporary) &&
            needComputeValues(where, temporary)) {
            values = ir::Var::make(temporaryAll.getName(),
                                   temporaryAll.getType().getDataType(),
                                   true, false);
            Expr size = getTemporarySize(where);
            Expr sizeAll = ir::Mul::make(size, ir::Call::make("omp_get_max_threads", {}, size.type()));

            // no decl needed for shared memory
            Stmt decl = Stmt();
            if ((isa<Forall>(where.getProducer()) && inParallelLoopDepth == 0)) {
                decl = VarDecl::make(values, ir::Literal::make(0));
            }
            Stmt allocate = Allocate::make(values, sizeAll);

            freeTemporary = Block::make(freeTemporary, Free::make(values));
            initializeTemporary = Block::make(decl, initializeTemporary, allocate);
        }
        /// Make a struct object that lowerAssignment and lowerAccess can read
        /// temporary value arrays from.
        TemporaryArrays arrays;
        arrays.values = values;
        this->temporaryArrays.insert({temporaryAll, arrays});

        return {initializeTemporary, freeTemporary};
    }

    vector<Stmt> LowererImplCUDA::codeToInitializeTemporary(Where where) {
        TensorVar temporary = where.getTemporary();

        const bool accelerateDense = canAccelerateDenseTemp(where).first;

        Stmt freeTemporary = Stmt();
        Stmt initializeTemporary = Stmt();
        if (isScalar(temporary.getType())) {
            initializeTemporary = defineScalarVariable(temporary, true);
            Expr tempSet = ir::Var::make(temporary.getName() + "_set", Datatype::Bool);
            Stmt initTempSet = VarDecl::make(tempSet, false);
            initializeTemporary = Block::make(initializeTemporary, initTempSet);
            tempToBitGuard[temporary] = tempSet;
        } else {
            // TODO: Need to support keeping track of initialized elements for
            //       temporaries that don't have sparse accelerator
            taco_iassert(!util::contains(guardedTemps, temporary) || accelerateDense);

            // When emitting code to accelerate dense workspaces with sparse iteration, we need the following arrays
            // to construct the result indices
            if(accelerateDense) {
                vector<Stmt> initAndFree = codeToInitializeDenseAcceleratorArrays(where);
                initializeTemporary = initAndFree[0];
                freeTemporary = initAndFree[1];
            }

            Expr values;
            if (util::contains(needCompute, temporary) &&
                needComputeValues(where, temporary)) {
                values = ir::Var::make(temporary.getName(),
                                       temporary.getType().getDataType(), true, false);

                Expr size = getTemporarySize(where);

                // no decl needed for shared memory
                Stmt decl = Stmt();
                if ((isa<Forall>(where.getProducer()) && inParallelLoopDepth == 0)) {
                    decl = VarDecl::make(values, ir::Literal::make(0));
                }
                Stmt allocate = Allocate::make(values, size);

                freeTemporary = Block::make(freeTemporary, Free::make(values));
                initializeTemporary = Block::make(decl, initializeTemporary, allocate);
            }

            /// Make a struct object that lowerAssignment and lowerAccess can read
            /// temporary value arrays from.
            TemporaryArrays arrays;
            arrays.values = values;
            this->temporaryArrays.insert({temporary, arrays});
        }
        return {initializeTemporary, freeTemporary};
    }
    Stmt LowererImplCUDA::initValues(Expr tensor, Expr initVal, Expr begin, Expr size) {
        Expr lower = simplify(ir::Mul::make(begin, size));
        Expr upper = simplify(ir::Mul::make(ir::Add::make(begin, 1), size));
        Expr p = Var::make("p" + util::toString(tensor), Int());
        Expr values = GetProperty::make(tensor, TensorProperty::Values);
        Stmt zeroInit = Store::make(values, p, initVal);
        LoopKind parallel = (isa<ir::Literal>(size) &&
                             to<ir::Literal>(size)->getIntValue() < (1 << 10))
                            ? LoopKind::Serial : LoopKind::Static_Chunked;
        if (util::contains(parallelUnitSizes, ParallelUnit::GPUBlock)) {
            return ir::VarDecl::make(ir::Var::make("status", Int()),
                                     ir::Call::make("cudaMemset", {values, ir::Literal::make(0, Int()),
                                                                   ir::Mul::make(ir::Sub::make(upper, lower),
                                                                                 ir::Literal::make(values.type().getNumBytes()))}, Int()));
        }
        return For::make(p, lower, upper, 1, zeroInit, parallel);
    }
    Stmt LowererImplCUDA::lowerForall(Forall forall)
    {
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
        for (const IndexVar& varToRecover : provGraph.newlyRecoverableParents(forall.getIndexVar(), definedIndexVars)) {
            // place pos guard
            if (forallNeedsUnderivedGuards && provGraph.isCoordVariable(varToRecover) &&
                provGraph.getChildren(varToRecover).size() == 1 &&
                provGraph.isPosVariable(provGraph.getChildren(varToRecover)[0])) {
                IndexVar posVar = provGraph.getChildren(varToRecover)[0];
                std::vector<ir::Expr> iterBounds = provGraph.deriveIterBounds(posVar, definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);

                Expr minGuard = Lt::make(indexVarToExprMap[posVar], iterBounds[0]);
                Expr maxGuard = Gte::make(indexVarToExprMap[posVar], iterBounds[1]);
                Expr guardCondition = Or::make(minGuard, maxGuard);
                if (isa<ir::Literal>(ir::simplify(iterBounds[0])) && ir::simplify(iterBounds[0]).as<ir::Literal>()->equalsScalar(0)) {
                    guardCondition = maxGuard;
                }
                ir::Stmt guard = ir::IfThenElse::make(guardCondition, ir::Continue::make());
                recoverySteps.push_back(guard);
            }

            Expr recoveredValue = provGraph.recoverVariable(varToRecover, definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
            taco_iassert(indexVarToExprMap.count(varToRecover));
            recoverySteps.push_back(VarDecl::make(indexVarToExprMap[varToRecover], recoveredValue));

            // After we've recovered this index variable, some iterators are now
            // accessible for use when declaring locator access variables. So, generate
            // the accessors for those locator variables as part of the recovery process.
            // This is necessary after a fuse transformation, for example: If we fuse
            // two index variables (i, j) into f, then after we've generated the loop for
            // f, all locate accessors for i and j are now available for use.
            std::vector<Iterator> itersForVar;
            for (auto& iters : iterators.levelIterators()) {
                // Collect all level iterators that have locate and iterate over
                // the recovered index variable.
                if (iters.second.getIndexVar() == varToRecover && iters.second.hasLocate()) {
                    itersForVar.push_back(iters.second);
                }
            }
            // Finally, declare all of the collected iterators' position access variables.
            recoverySteps.push_back(this->declLocatePosVars(itersForVar));

            // place underived guard
            std::vector<ir::Expr> iterBounds = provGraph.deriveIterBounds(varToRecover, definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
            if (forallNeedsUnderivedGuards && underivedBounds.count(varToRecover) &&
                !provGraph.hasPosDescendant(varToRecover)) {

                // FIXME: [Olivia] Check this with someone
                // Removed underived guard if indexVar is bounded is divisible by its split child indexVar
                vector<IndexVar> children = provGraph.getChildren(varToRecover);
                bool hasDirectDivBound = false;
                std::vector<ir::Expr> iterBoundsInner = provGraph.deriveIterBounds(forall.getIndexVar(), definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
                for (auto& c: children) {
                    if (provGraph.hasExactBound(c) &&
                        provGraph.derivationPath(varToRecover, c).size() == 2) {
                        const auto iterBoundsUnderivedChild =
                                provGraph.deriveIterBounds(c, definedIndexVarsOrdered,
                                                           underivedBounds, indexVarToExprMap,
                                                           iterators);
                        if (iterBoundsUnderivedChild[1].as<ir::Literal>()->getValue<int>() %
                            iterBoundsInner[1].as<ir::Literal>()->getValue<int>() == 0) {
                            hasDirectDivBound = true;
                            break;
                        }
                    }
                }
                if (!hasDirectDivBound) {
                    Stmt guard = IfThenElse::make(Gte::make(indexVarToExprMap[varToRecover],
                                                            underivedBounds[varToRecover][1]),
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
                auto bounds = this->provGraph.deriveIterBounds(inner, definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
                // Use the difference between the bounds to find the size of the loop.
                auto dimLen = ir::Sub::make(bounds[1], bounds[0]);
                // For a variable f divided into into f1 and f2, the guard ensures that
                // for iteration f, f should be within f1 * dimLen and (f1 + 1) * dimLen.
                auto guard = ir::Gte::make(this->indexVarToExprMap[varToRecover], ir::Mul::make(ir::Add::make(this->indexVarToExprMap[outer], 1), dimLen));
                recoverySteps.push_back(IfThenElse::make(guard, ir::Continue::make()));
            }
        }
        Stmt recoveryStmt = Block::make(recoverySteps);

        taco_iassert(!definedIndexVars.count(forall.getIndexVar()));
        definedIndexVars.insert(forall.getIndexVar());
        definedIndexVarsOrdered.push_back(forall.getIndexVar());

        if (forall.getParallelUnit() != ParallelUnit::NotParallel) {
            taco_iassert(!parallelUnitSizes.count(forall.getParallelUnit()));
            taco_iassert(!parallelUnitIndexVars.count(forall.getParallelUnit()));
            parallelUnitIndexVars[forall.getParallelUnit()] = forall.getIndexVar();
            vector<Expr> bounds = provGraph.deriveIterBounds(forall.getIndexVar(), definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);
            parallelUnitSizes[forall.getParallelUnit()] = ir::Sub::make(bounds[1], bounds[0]);
        }

        MergeLattice caseLattice = MergeLattice::make(forall, iterators, provGraph, definedIndexVars, whereTempsToResult);
        vector<Access> resultAccesses;
        set<Access> reducedAccesses;
        std::tie(resultAccesses, reducedAccesses) = getResultAccesses(forall);

        // Pre-allocate/initialize memory of value arrays that are full below this
        // loops index variable
        Stmt preInitValues = initResultArrays(forall.getIndexVar(), resultAccesses,
                                              reducedAccesses);

        // Emit temporary initialization if forall is sequential or parallelized by
        // cpu threads and leads to a where statement
        // This is for workspace hoisting by 1-level
        vector<Stmt> temporaryValuesInitFree = {Stmt(), Stmt()};
        auto temp = temporaryInitialization.find(forall);
        if (temp != temporaryInitialization.end() && forall.getParallelUnit() ==
                                                     ParallelUnit::NotParallel && !isScalar(temp->second.getTemporary().getType()))
            temporaryValuesInitFree = codeToInitializeTemporary(temp->second);
        else if (temp != temporaryInitialization.end() && forall.getParallelUnit() ==
                                                          ParallelUnit::CPUThread && !isScalar(temp->second.getTemporary().getType())) {
            temporaryValuesInitFree = codeToInitializeTemporaryParallel(temp->second, forall.getParallelUnit());
        }

        Stmt loops;
        // Emit a loop that iterates over over a single iterator (optimization)
        if (caseLattice.iterators().size() == 1 && caseLattice.iterators()[0].isUnique()) {
            MergeLattice loopLattice = caseLattice.getLoopLattice();

            MergePoint point = loopLattice.points()[0];
            Iterator iterator = loopLattice.iterators()[0];

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
            bool canAccelWithSparseIteration =
                    provGraph.isFullyDerived(iterator.getIndexVar()) &&
                    iterator.isDimensionIterator() && locators.size() == 1;
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

            if (!isWhereProducer && hasPosDescendant && underivedAncestors.size() > 1 && provGraph.isPosVariable(iterator.getIndexVar()) && posDescendant == forall.getIndexVar()) {
                loops = lowerForallFusedPosition(forall, iterator, locators, inserters, appenders, caseLattice,
                                                 reducedAccesses, recoveryStmt);
            }
            else if (canAccelWithSparseIteration) {
                loops = lowerForallDenseAcceleration(forall, locators, inserters, appenders, caseLattice, reducedAccesses, recoveryStmt);
            }
                // Emit dimension coordinate iteration loop
            else if (iterator.isDimensionIterator()) {
                loops = lowerForallDimension(forall, point.locators(), inserters, appenders, caseLattice,
                                             reducedAccesses, recoveryStmt);
            }
                // Emit position iteration loop
            else if (iterator.hasPosIter()) {
                loops = lowerForallPosition(forall, iterator, locators, inserters, appenders, caseLattice,
                                            reducedAccesses, recoveryStmt);
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
            loops = lowerMergeLattice(caseLattice, underivedAncestors[0],
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
        if (forall.getParallelUnit() != ParallelUnit::NotParallel) {
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

    Stmt LowererImplCUDA::lowerForallCloned(Forall forall) {
        // want to emit guards outside of loop to prevent unstructured loop exits

        // construct guard
        // underived or pos variables that have a descendant that has not been defined yet
        vector<IndexVar> varsWithGuard;
        for (auto var : provGraph.getAllIndexVars()) {
            if (provGraph.isRecoverable(var, definedIndexVars)) {
                continue; // already recovered
            }
            if (provGraph.isUnderived(var) && !provGraph.hasPosDescendant(var)) { // if there is pos descendant then will be guarded already
                varsWithGuard.push_back(var);
            }
            else if (provGraph.isPosVariable(var)) {
                // if parent is coord then this is variable that will be guarded when indexing into coord array
                if(provGraph.getParents(var).size() == 1 && provGraph.isCoordVariable(provGraph.getParents(var)[0])) {
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
        Expr maxOffset = 0;
        bool setMaxOffset = false;

        for (auto var : varsWithGuard) {
            std::vector<IndexVar> currentDefinedVarOrder = definedIndexVarsOrdered; // TODO: get defined vars at time of this recovery

            std::map<IndexVar, Expr> minChildValues = indexVarToExprMap;
            std::map<IndexVar, Expr> maxChildValues = indexVarToExprMap;

            for (auto child : provGraph.getFullyDerivedDescendants(var)) {
                if (!definedIndexVars.count(child)) {
                    std::vector<ir::Expr> childBounds = provGraph.deriveIterBounds(child, currentDefinedVarOrder, underivedBounds, indexVarToExprMap, iterators);

                    minChildValues[child] = childBounds[0];
                    maxChildValues[child] = childBounds[1];

                    // recover new parents
                    for (const IndexVar& varToRecover : provGraph.newlyRecoverableParents(child, definedForGuard)) {
                        Expr recoveredValue = provGraph.recoverVariable(varToRecover, definedIndexVarsOrdered, underivedBounds,
                                                                        minChildValues, iterators);
                        Expr maxRecoveredValue = provGraph.recoverVariable(varToRecover, definedIndexVarsOrdered, underivedBounds,
                                                                           maxChildValues, iterators);
                        if (!setMaxOffset) { // TODO: work on simplifying this
                            maxOffset = ir::Add::make(maxOffset, ir::Sub::make(maxRecoveredValue, recoveredValue));
                            setMaxOffset = true;
                        }
                        taco_iassert(indexVarToExprMap.count(varToRecover));

                        guardRecoverySteps.push_back(VarDecl::make(indexVarToExprMap[varToRecover], recoveredValue));
                        definedForGuard.insert(varToRecover);
                    }
                    definedForGuard.insert(child);
                }
            }

            minVarValues[var] = provGraph.recoverVariable(var, currentDefinedVarOrder, underivedBounds, minChildValues, iterators);
            maxVarValues[var] = provGraph.recoverVariable(var, currentDefinedVarOrder, underivedBounds, maxChildValues, iterators);
        }

        // Build guards
        Expr guardCondition;
        for (auto var : varsWithGuard) {
            std::vector<ir::Expr> iterBounds = provGraph.deriveIterBounds(var, definedIndexVarsOrdered, underivedBounds, indexVarToExprMap, iterators);

            Expr minGuard = Lt::make(minVarValues[var], iterBounds[0]);
            Expr maxGuard = Gte::make(ir::Add::make(maxVarValues[var], ir::simplify(maxOffset)), iterBounds[1]);
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

        taco_uassert(guardCondition.defined())
                << "Unable to vectorize or unroll loop over unbound variable " << forall.getIndexVar();

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
    Stmt LowererImplCUDA::lowerWhere(Where where) {
        TensorVar temporary = where.getTemporary();
        bool accelerateDenseWorkSpace, sortAccelerator;
        std::tie(accelerateDenseWorkSpace, sortAccelerator) =
                canAccelerateDenseTemp(where);

        // Declare and initialize the where statement's temporary
        vector<Stmt> temporaryValuesInitFree = {Stmt(), Stmt()};
        bool temporaryHoisted = false;
        for (auto it = temporaryInitialization.begin(); it != temporaryInitialization.end(); ++it) {
            if (it->second == where && it->first.getParallelUnit() ==
                                       ParallelUnit::NotParallel && !isScalar(temporary.getType())) {
                temporaryHoisted = true;
            }
        }

        if (!temporaryHoisted) {
            temporaryValuesInitFree = codeToInitializeTemporary(where);
        }

        Stmt initializeTemporary = temporaryValuesInitFree[0];
        Stmt freeTemporary = temporaryValuesInitFree[1];

        match(where.getConsumer(),
              std::function<void(const AssignmentNode*)>([&](const AssignmentNode* op) {
                  if (op->lhs.getTensorVar().getOrder() > 0) {
                      whereTempsToResult[where.getTemporary()] = (const AccessNode *) op->lhs.ptr;
                  }
              })
        );

        Stmt consumer = lower(where.getConsumer());
        if (accelerateDenseWorkSpace && sortAccelerator) {
            // We need to sort the indices array
            Expr listOfIndices = tempToIndexList.at(temporary);
            Expr listOfIndicesSize = tempToIndexListSize.at(temporary);
            Expr sizeOfElt = ir::Sizeof::make(listOfIndices.type());
            Stmt sortCall = ir::Sort::make({listOfIndices, listOfIndicesSize, sizeOfElt});
            consumer = Block::make(sortCall, consumer);
        }

        // Now that temporary allocations are hoisted, we always need to emit an initialization loop before entering the
        // producer but only if there is no dense acceleration
        if (util::contains(needCompute, temporary) && !isScalar(temporary.getType()) && !accelerateDenseWorkSpace) {
            // TODO: We only actually need to do this if:
            //      1) We use the temporary multiple times
            //      2) The PRODUCER RHS is sparse(not full). (Guarantees that old values are overwritten before consuming)

            Expr p = Var::make("p" + temporary.getName(), Int());
            Expr values = ir::Var::make(temporary.getName(),
                                        temporary.getType().getDataType(),
                                        true, false);
            Expr size = getTemporarySize(where);
            Stmt zeroInit = Store::make(values, p, ir::Literal::zero(temporary.getType().getDataType()));
            Stmt loopInit = For::make(p, 0, size, 1, zeroInit, LoopKind::Serial);
            initializeTemporary = Block::make(initializeTemporary, loopInit);
        }

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
        if (accelerateDenseWorkSpace) {
            const Expr indexListSizeExpr = tempToIndexListSize.at(temporary);
            const Stmt indexListSizeDecl = VarDecl::make(indexListSizeExpr, ir::Literal::make(0));
            initializeTemporary = Block::make(indexListSizeDecl, initializeTemporary);
        }

        if (restoreAtomicDepth) {
            markAssignsAtomicDepth++;
        }

        whereConsumers.pop_back();
        whereTemps.pop_back();
        whereTempsToResult.erase(where.getTemporary());
        return Block::make(initializeTemporary, producer, markAssignsAtomicDepth > 0 ? capturedLocatePos : ir::Stmt(), consumer,  freeTemporary);
    }



}