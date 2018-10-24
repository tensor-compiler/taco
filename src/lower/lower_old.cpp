#include "taco/lower/lower.h"

#include "taco/index_notation/index_notation.h"

#include "taco/ir/ir.h"
#include "taco/ir/ir_visitor.h"
#include "taco/ir/simplify.h"
#include "ir/ir_generators.h"

#include "lower_codegen.h"
#include "iterators.h"
#include "tensor_path.h"
#include "merge_lattice_old.h"
#include "iteration_graph.h"
#include "expr_tools.h"
#include "taco/lower/iterator.h"
#include "taco/index_notation/index_notation_nodes.h"
#include "taco/index_notation/index_notation_rewriter.h"
#include "taco/index_notation/schedule.h"
#include "error/error_checks.h"
#include "taco/error/error_messages.h"
#include "taco/util/name_generator.h"
#include "taco/util/collections.h"
#include "taco/util/strings.h"

using namespace std;
using namespace taco::ir;


namespace taco {
namespace old {

struct Ctx {
  /// Determines what kind of code to emit (e.g. compute and/or assembly)
  std::set<Property>       properties;

  /// The iteration graph to use for lowering the index expression
  IterationGraph           iterationGraph;

  /// The iterators of the tensor tree levels
  Iterators                iterators;

  /// Maps tensor (scalar) temporaries to IR variables.
  /// (Not clear if this approach to temporaries is too hacky.)
  std::map<TensorVar,Expr> temporaries;

  std::map<Iterator,Expr>  idxVars;

  Expr                     valsCapacity;

  Ctx(const IterationGraph& iterationGraph,
          const set<Property>& properties,
          const map<TensorVar,Expr>& tensorVars) {
    this->properties = properties;
    this->iterationGraph = iterationGraph;
    this->iterators = Iterators(iterationGraph, tensorVars);
  }
};

struct Target {
  Expr tensor;
  Expr pos;
};

enum ComputeCase {
  // Emit the last free variable. We first recurse to compute remaining
  // reduction variables into a temporary, before we compute and store the
  // main expression
  LAST_FREE,

  // Emit a variable above the last free variable. First emit code to compute
  // available expressions and store them in temporaries, before
  // we recurse on the next index variable.
  ABOVE_LAST_FREE,

  // Emit a variable below the last free variable. First recurse to emit
  // remaining (summation) variables, before we add in the available expressions
  // for the current summation variable.
  BELOW_LAST_FREE
};

static ComputeCase getComputeCase(const IndexVar& indexVar,
                                  const IterationGraph& iterationGraph) {
  if (iterationGraph.isLastFreeVariable(indexVar)) {
    return LAST_FREE;
  }
  else if (iterationGraph.hasFreeVariableDescendant(indexVar)) {
    return ABOVE_LAST_FREE;
  }
  else {
    return BELOW_LAST_FREE;
  }
}

static bool needsZero(const Ctx& ctx,
                      const std::vector<IndexVar>& resultIdxVars) {
  const auto& resultTensorPath = ctx.iterationGraph.getResultTensorPath();

  for (const auto& idxVar : resultIdxVars) {
    if (ctx.iterators[resultTensorPath.getStep(idxVar)].hasInsert()) {
      for (const auto& tensorPath : ctx.iterationGraph.getTensorPaths()) {
        if (util::contains(tensorPath.getVariables(), idxVar) &&
            !ctx.iterators[tensorPath.getStep(idxVar)].isFull()) {
          return true;
        }
      }
    }
  }

  return false;
}

static bool needsZero(const Ctx& ctx) {
  const auto& graph = ctx.iterationGraph;
  const auto& resultIdxVars = graph.getResultTensorPath().getVariables();

  if (graph.hasReductionVariableAncestor(resultIdxVars.back())) {
    return true;
  }

  return needsZero(ctx, resultIdxVars);
}

static IndexExpr emitAvailableExprs(const IndexVar& indexVar,
                                    const IndexExpr& indexExpr, Ctx* ctx,
                                    vector<Stmt>* stmts) {
  vector<IndexVar>  visited    = ctx->iterationGraph.getAncestors(indexVar);
  vector<IndexExpr> availExprs = getAvailableExpressions(indexExpr, visited);
  map<IndexExpr,IndexExpr> substitutions;
  for (const IndexExpr& availExpr : availExprs) {
    TensorVar t("t" + indexVar.getName(), availExpr.getDataType());
    substitutions.insert({availExpr, taco::Access(t)});
    Expr tensorVarExpr = Var::make(t.getName(), availExpr.getDataType());
    ctx->temporaries.insert({t, tensorVarExpr});
    Expr expr = lowerToScalarExpression(availExpr, ctx->iterators,
                                        ctx->iterationGraph, ctx->temporaries);
    stmts->push_back(VarDecl::make(tensorVarExpr, expr));
  }
  return replace(indexExpr, substitutions);
}

static void emitComputeExpr(const Target& target, const IndexVar& indexVar,
                            const IndexExpr& indexExpr, const Ctx& ctx,
                            vector<Stmt>* stmts, bool accum) {
  Expr expr = lowerToScalarExpression(indexExpr, ctx.iterators,
                                      ctx.iterationGraph, ctx.temporaries);
  auto& iterationGraph = ctx.iterationGraph;
  if (target.pos.defined()) {
    Stmt store = iterationGraph.hasReductionVariableAncestor(indexVar) || accum
        ? compoundStore(target.tensor, target.pos, expr)
        :   Store::make(target.tensor, target.pos, expr);
    stmts->push_back(store);
  }
  else {
    Stmt assign = iterationGraph.hasReductionVariableAncestor(indexVar) || accum
        ?  compoundAssign(target.tensor, expr)
        : Assign::make(target.tensor, expr);
    stmts->push_back(assign);
  }
}

static LoopKind doParallelize(const IndexVar& indexVar, const Expr& tensor,
                              const Ctx& ctx) {
  if (ctx.iterationGraph.getAncestors(indexVar).size() != 1 ||
      ctx.iterationGraph.isReduction(indexVar) ||
      util::contains(ctx.properties, Assemble)) {
    return LoopKind::Serial;
  }

  const TensorPath& resultPath = ctx.iterationGraph.getResultTensorPath();
  for (size_t i = 0; i < resultPath.getSize(); i++){
    if (!ctx.iterators[resultPath.getStep(i)].hasInsert()) {
      return LoopKind::Serial;
    }
  }

  const TensorPath parallelizedAccess = [&]() {
    const auto tensorName = tensor.as<Var>()->name;
    for (const auto& tensorPath : ctx.iterationGraph.getTensorPaths()) {
      if (tensorPath.getAccess().getTensorVar().getName() == tensorName) {
        return tensorPath;
      }
    }
    taco_iassert(false);
    return TensorPath();
  }();

  if (parallelizedAccess.getSize() <= 2) {
    return LoopKind::Static;
  }

  for (size_t i = 1; i < parallelizedAccess.getSize(); ++i) {
    if (ctx.iterators[parallelizedAccess.getStep(i)].isFull()) {
      return LoopKind::Static;
    }
  }

  return LoopKind::Dynamic;
}

/// Expression evaluates to true iff none of the iteratators are exhausted
static Expr noneExhausted(const std::vector<Iterator>& iterators) {
  taco_iassert(!iterators.empty());

  std::vector<Expr> stepIterLqEnd;
  for (const auto& iter : iterators) {
    if (!iter.isFull()) {
      Expr iterUnexhausted = Lt::make(iter.getIteratorVar(), iter.getEndVar());
      stepIterLqEnd.push_back(iterUnexhausted);
    }
  }
  return !stepIterLqEnd.empty() ? conjunction(stepIterLqEnd) :
         Lt::make(iterators[0].getIteratorVar(), iterators[0].getEndVar());
}

/// Expression evaluates to true iff all the iterator idx vars are equal to idx
/// or if there are no iterators.
static Expr allEqualTo(const std::vector<Iterator>& iterators, Expr idx) {
  if (iterators.empty()) {
    return true;
  }

  std::vector<Expr> iterIdxEqualToIdx;
  for (const auto& iter : iterators) {
    iterIdxEqualToIdx.push_back(Eq::make(iter.getCoordVar(), idx));
  }
  return conjunction(iterIdxEqualToIdx);
}

static Expr allValidDerefs(const std::vector<Iterator>& iterators,
                           const std::set<Iterator>& guardedIters) {
  std::vector<Expr> iterValid;
  for (const auto& iter : iterators) {
    if (util::contains(guardedIters, iter)) {
      iterValid.push_back(iter.getValidVar());
    }
  }
  return iterValid.empty() ? true : conjunction(iterValid);
}

/// Returns a bitmask where the i-th bit is set to true iff the i-th iterator in
/// `iterators` is contained in `selected`.
static Expr indicatorMask(const vector<Iterator>& iterators,
                          const vector<Iterator>& selected) {
  unsigned long long mask = 0;
  for (unsigned long long i = 0, b = 1;
       i < iterators.size(); ++i, b *= 2) {
    mask |= b * (contains(selected, iterators[i]));
  }
  return mask;
}

///// Returns the iterator for the `idx` variable from `iterators`, or Iterator()
///// none of the iterator iterate over `idx`.
//static Iterator getIterator(const Expr& idx,
//                            const vector<Iterator>& iterators) {
//  for (auto& iterator : iterators) {
//    if (iterator.getIdxVar() == idx) {
//      return iterator;
//    }
//  }
//  return Iterator();
//}

static vector<Iterator> removeIterator(const Expr& idx,
                                       const vector<Iterator>& iterators) {
  vector<Iterator> result;
  for (auto& iterator : iterators) {
    if (iterator.getCoordVar() != idx) {
      result.push_back(iterator);
    }
  }
  return result;
}

static Stmt createIfStatements(const vector<pair<Expr,Stmt>> &cases,
                               const MergeLattice& lattice,
                               const Expr switchExpr) {
  if (cases.size() == 1 && isa<ir::Literal>(cases[0].first) &&
      to<ir::Literal>(cases[0].first)->getValue<bool>()) {
    return cases[0].second;
  }

  vector<pair<Expr,Stmt>> ifCases;
  pair<Expr,Stmt> elseCase;
  for (auto& cas : cases) {
    auto lit = cas.first.as<ir::Literal>();
    if (lit != nullptr && lit->type == Bool && lit->getValue<bool>() == 1) {
      taco_iassert(!elseCase.first.defined()) <<
          "there should only be one true case";
      elseCase = cas;
    }
    else {
      ifCases.push_back(cas);
    }
  }

  if (elseCase.first.defined()) {
    ifCases.push_back(elseCase);
    return Case::make(ifCases, true);
  }

  return switchExpr.defined()
         ? Switch::make(ifCases, switchExpr)
         : Case::make(ifCases, lattice.isFull());
}

static std::vector<Expr> getIdxVars(std::map<Iterator,Expr>& idxVars,
                                    const Iterator lastIterator,
                                    const bool includeLastIdxVar = true) {
  std::vector<Expr> ret;

  taco_iassert(lastIterator.defined());
  if (includeLastIdxVar) {
    ret.push_back(idxVars[lastIterator]);
  }

  for (Iterator iter = lastIterator.getParent(); iter.defined();
       iter = iter.getParent()) {
    ret.push_back(idxVars[iter]);
  }
  std::reverse(ret.begin(), ret.end());

  return ret;
}

/// Lowers an index expression to imperative code according to the loop ordering
/// described by an iteration graph. This  algorithm was first outlined in paper
/// "The Tensor Algebra Compiler", but has since been generalized.
static vector<Stmt> lower(const Target&      target,
                          const IndexVar&    indexVar,
                          IndexExpr          indexExpr,
                          const set<Access>& exhausted,
                          Ctx&           ctx) {
  IterationGraph iterationGraph = ctx.iterationGraph;

  MergeLattice lattice = MergeLattice::make(indexExpr, indexVar,
                                            ctx.iterationGraph,
                                            ctx.iterators);
  const auto& latticeRangeIterators = lattice.getRangeIterators();

  TensorPath     resultPath     = iterationGraph.getResultTensorPath();
  TensorPathStep resultStep     = resultPath.getStep(indexVar);
  Iterator       resultIterator = (resultStep.getPath().defined())
                                  ? ctx.iterators[resultStep]
                                  : Iterator();

  const bool accumulate   = util::contains(ctx.properties, Accumulate);
  const bool emitCompute  = util::contains(ctx.properties, Compute);
  const bool emitAssemble = util::contains(ctx.properties, Assemble);

  // Emit while loops to merge inputs if need to co-iterate two or more inputs
  // or if deduplication is needed.
  const bool emitMerge = (latticeRangeIterators.size() > 1) ||
                         !latticeRangeIterators[0].isUnique();

  std::vector<Stmt> code;

  // Emit code to initialize pos variables:
  // B2_pos = B2_pos_arr[B1_pos];
  ModeFunction iterFunc;
  for (auto& iterator : latticeRangeIterators) {
    if (iterator.hasPosIter()) {
      iterFunc = iterator.posBounds();
    } else {
      taco_iassert(iterator.hasCoordIter());
      auto coords = getIdxVars(ctx.idxVars, iterator, false);
      iterFunc = iterator.coordBounds(coords);
      taco_iassert(iterFunc.defined());
    }

    if (iterFunc.compute().defined()) {
      code.push_back(iterFunc.compute());
    }
    if (emitMerge) {
      Expr iterVar = iterator.getIteratorVar();
      Stmt initIter = VarDecl::make(iterVar, iterFunc.getResults()[0]);
      Stmt initEnd = VarDecl::make(iterator.getEndVar(),
                                   iterFunc.getResults()[1]);
      code.push_back(initIter);
      code.push_back(initEnd);
    }
  }

  if (emitAssemble && resultIterator.defined()) {
    if (resultIterator.hasAppend() && !resultIterator.isBranchless()) {
      Expr begin = resultIterator.getBeginVar();
      Stmt initBegin = VarDecl::make(begin, resultIterator.getPosVar());
      code.push_back(initBegin);
    }

    if (resultIterator.getParent().hasAppend() ||
        resultStep == resultPath.getStep(0)) {
      Expr resultParentPos = resultIterator.getParent().getPosVar();
      Expr initBegin = resultParentPos;
      Expr initEnd = simplify(ir::Add::make(resultParentPos, 1ll));

      TensorPathStep initStep = resultStep;
      Iterator initIterator = resultIterator;
      while (initIterator.defined() && initIterator.hasInsert()) {
        Expr size = initIterator.getSize();
        initBegin = simplify(ir::Mul::make(initBegin, size));
        initEnd = simplify(ir::Mul::make(initEnd, size));

        Stmt initCoords = initIterator.getInsertInitCoords(initBegin, initEnd);
        if (initCoords.defined()) {
          code.push_back(initCoords);
        }

        if (initStep == resultPath.getLastStep()) {
          initIterator = Iterator();
        } else {
          initStep = resultPath.getStep(initStep.getStep() + 1);
          initIterator = ctx.iterators[initStep];
        }
      }

      if (initIterator.defined()) {
        taco_iassert(initIterator.hasAppend());
        Stmt initEdges = initIterator.getAppendInitEdges(initBegin, initEnd);
        if (initEdges.defined()) {
          code.push_back(initEdges);
        }
      } else if (emitCompute && resultStep != resultPath.getStep(0)) {
        Expr resultTensor = resultIterator.getTensor();
        Expr vals = GetProperty::make(resultTensor, TensorProperty::Values);

        Expr newCapacity = ir::Mul::make(2ll, initEnd);
        Stmt resizeVals = Allocate::make(vals, newCapacity, true);
        Stmt updateCapacity = Assign::make(ctx.valsCapacity, newCapacity);

        Expr shouldResize = Lte::make(ctx.valsCapacity, initEnd);
        Stmt resizeBody = Block::make({resizeVals, updateCapacity});
        Stmt maybeResizeVals = IfThenElse::make(shouldResize, resizeBody);
        code.push_back(maybeResizeVals);

        const auto& resultIdxVars = resultPath.getVariables();
        const auto it = std::find(resultIdxVars.begin(),
                                  resultIdxVars.end(), indexVar);
        const std::vector<IndexVar> nextIdxVars(it, resultIdxVars.end());
        if (needsZero(ctx, nextIdxVars)) {
          const std::string iterName = "p" + resultTensor.as<Var>()->name;
          Expr iterVar = Var::make(iterName, Int());
          Expr zero = ir::Literal::zero(target.tensor.type());
          Stmt zeroStmt = Store::make(target.tensor, iterVar, zero);
          Stmt zeroLoop = For::make(iterVar, initBegin, initEnd, 1ll, zeroStmt);
          code.push_back(zeroLoop);
        }
      }
    }
  }

  // Emit one loop per lattice point lp
  std::vector<Stmt> loops;
  for (MergePoint lp : lattice.getPoints()) {
    MergeLattice lpLattice = lattice.getSubLattice(lp);

    const std::vector<Iterator>& lpIterators = lp.getIterators();
    const std::vector<Iterator>& lpRangeIterators = lp.getRangers();

    const std::vector<Iterator> lpLocateIterators = util::remove(
        lpIterators, lpRangeIterators);

    std::vector<Stmt> loopBody;
    std::set<Iterator> guardedIters;

    // Emit code to initialize sequential access idx variables:
    // int kB = B1_idx_arr[B1_pos];
    // int kc = c0_idx_arr[c0_pos];
    for (auto& iterator : lpRangeIterators) {
      ModeFunction access;
      if (iterator.hasPosIter()) {
        const auto coords = getIdxVars(ctx.idxVars, iterator, false);
        access = iterator.posAccess(coords);
      } else {
        Expr coord = iterator.getCoordVar();
        auto idxVars = util::combine(getIdxVars(ctx.idxVars, iterator, false),
                                     {coord});
        access = iterator.coordAccess(idxVars);
      }
      Expr deref = access.getResults()[0];
      Expr valid = access.getResults()[1];

      Stmt initDerived = VarDecl::make(iterator.getDerivedVar(),
                                       simplify(deref));

      if (iterFunc.compute().defined()) {
        loopBody.push_back(iterFunc.compute());
      }
      loopBody.push_back(initDerived);
      if (!isa<ir::Literal>(valid)) {
        Stmt initValid = VarDecl::make(iterator.getValidVar(), valid);
        loopBody.push_back(initValid);
        guardedIters.insert(iterator);
      } else {
        taco_iassert(isValue(valid, true));
      }
    }

    std::vector<Stmt> mergeCode;

    const bool mergeWithSwitch =
        (lpRangeIterators.size() > 2 &&
         lpRangeIterators.size() <= (size_t)UInt().getNumBits() &&
         lpLattice.getSize() == (1u << lpRangeIterators.size()) - 1);

    // Emit code to initialize the index variable:
    // k = min(kB, kc);
    Expr idx, ind;
    if (mergeWithSwitch) {
      std::tie(idx, ind) = minWithIndicator(indexVar.getName(),
                                            lpRangeIterators, &mergeCode);
    } else {
      idx = min(indexVar.getName(), lpRangeIterators, &mergeCode);
    }

    // Associate merged index variable with merged iterators
    for (auto& iterator : lpIterators) {
      ctx.idxVars[iterator] = idx;
    }
    if (resultIterator.defined()) {
      ctx.idxVars[resultIterator] = idx;
    }

    // Emit code to initialize random access pos variables:
    // D1_pos = (D0_pos * 3) + k;
    for (size_t i = 0; i < lpLocateIterators.size() +
         (resultIterator.defined() && resultIterator.hasInsert()); ++i) {
      Iterator iterator = (i == lpLocateIterators.size()) ? resultIterator :
                          lpLocateIterators[i];

      auto coords = getIdxVars(ctx.idxVars, iterator, true);
      ModeFunction locate = iterator.locate(coords);
      Stmt initPos = VarDecl::make(iterator.getPosVar(),
                                   simplify(locate.getResults()[0]));

      if (locate.compute().defined()) {
        mergeCode.push_back(locate.compute());
      }
      mergeCode.push_back(initPos);
      if (!isa<ir::Literal>(locate.getResults()[1]) && iterator != resultIterator) {
        Stmt initValid = VarDecl::make(iterator.getValidVar(),
                                       locate.getResults()[1]);

        mergeCode.push_back(initValid);
        guardedIters.insert(iterator);
      } else {
        taco_iassert(iterator == resultIterator ||
                     (locate.getResults()[1].type().isBool() &&
                      to<ir::Literal>(locate.getResults()[1])->getValue<bool>()));
      }
    }

    for (auto& iterator : lpRangeIterators) {
      if (iterator.hasPosIter() && !iterator.isUnique()) {
        Expr segendVar = iterator.getSegendVar();
        Expr nextPos = ir::Add::make(iterator.getPosVar(), 1ll);
        Stmt initSegend = VarDecl::make(segendVar, nextPos);
        mergeCode.push_back(initSegend);
      }
    }

    // Emit code to resize vals array when simultaneously performing assembly
    // and compute and result components are appended
    Stmt maybeResizeVals;
    if (emitCompute && emitAssemble && resultIterator.defined() &&
        resultIterator.hasAppend() && resultStep == resultPath.getLastStep()) {
      Expr resultTensor = resultIterator.getTensor();
      Expr vals = GetProperty::make(resultTensor, TensorProperty::Values);

      Expr resultPos = resultIterator.getPosVar();
      Expr newValsEnd = ir::Add::make(resultPos, 1ll);
      Expr newCapacity = ir::Mul::make(2ll, newValsEnd);
      Stmt resizeVals = Allocate::make(vals, newCapacity, true);
      Stmt updateCapacity = Assign::make(ctx.valsCapacity, newCapacity);
      Stmt doResize = Block::make({resizeVals, updateCapacity});

      Expr shouldResize = Lte::make(ctx.valsCapacity, newValsEnd);
      maybeResizeVals = IfThenElse::make(shouldResize, doResize);
    }
    if (maybeResizeVals.defined() && lpLattice.getSize() > 1) {
      mergeCode.push_back(maybeResizeVals);
    }

    // Emit one case per lattice point in the sub-lattice rooted at lp
    std::vector<std::pair<Expr,Stmt>> cases;
    for (MergePoint lq : lpLattice.getPoints()) {
      const std::vector<Iterator>& lqIterators = lq.getIterators();
      const std::vector<Iterator>& lqRangeIterators = lq.getRangers();
      const std::vector<Iterator> lqLocateIterators = util::remove(
          lqIterators, lqRangeIterators);

      IndexExpr lqexpr = lq.getExpr();
      std::set<Access> exhausted = exhaustedAccesses(lq, lattice);

      std::vector<Stmt> caseBody;

      if (maybeResizeVals.defined() && lpLattice.getSize() == 1) {
        caseBody.push_back(maybeResizeVals);
      }

      // Emit compute code for three cases: above, at or below the last free var
      ComputeCase ivarCase = getComputeCase(indexVar, iterationGraph);

      // Emit available sub-expressions at this loop level
      if (emitCompute && ABOVE_LAST_FREE == ivarCase) {
        lqexpr = emitAvailableExprs(indexVar, lqexpr, &ctx, &caseBody);
      }

      if (iterationGraph.getChildren(indexVar).size() == 1) {
        // Recursive call to emit iteration graph children
        for (auto& child : iterationGraph.getChildren(indexVar)) {
          IndexExpr childExpr = lqexpr;
          Target childTarget = target;
          if (ivarCase == LAST_FREE || ivarCase == BELOW_LAST_FREE) {
            // Extract the expression to compute at the next level. If there's no
            // computation on the next level (for this lattice case) then skip it.
            childExpr = getSubExprOld(lqexpr, iterationGraph.getDescendants(child));
            if (!childExpr.defined()) continue;

            // Reduce child expression into temporary
            TensorVar t("t" + child.getName(), childExpr.getDataType());
            Expr tensorVarExpr = Var::make(t.getName(), childExpr.getDataType());
            ctx.temporaries.insert({t, tensorVarExpr});
            childTarget.tensor = tensorVarExpr;
            childTarget.pos    = Expr();
            if (emitCompute) {
              Expr zero = ir::Literal::zero(tensorVarExpr.type());
              caseBody.push_back(VarDecl::make(tensorVarExpr, zero));
            }

            // Rewrite lqExpr to substitute the expression computed at the next
            // level with the temporary
            lqexpr = replace(lqexpr, {{childExpr,taco::Access(t)}});
          }
          auto childCode = lower(childTarget, child, childExpr, exhausted, ctx);
          util::append(caseBody, childCode);
        }

        // Emit code to compute and store/assign result
        if (emitCompute &&
            (ivarCase == LAST_FREE || ivarCase == BELOW_LAST_FREE)) {
          emitComputeExpr(target, indexVar, lqexpr, ctx, &caseBody, accumulate);
        }
      }
      else {
        // Recursive call to emit iteration graph children
        vector<IndexExpr> childVars;
        for (auto& child : iterationGraph.getChildren(indexVar)) {
          IndexExpr childExpr = lqexpr;
          Target childTarget = target;
          if (ivarCase == LAST_FREE || ivarCase == BELOW_LAST_FREE) {
            // Extract the expression to compute at the next level. If there's no
            // computation on the next level (for this lattice case) then skip it.
            childExpr = getSubExpr(lqexpr, iterationGraph.getDescendants(child));
            if (!childExpr.defined()) continue;

            // Reduce child expression into temporary
            TensorVar t("t" + child.getName(), childExpr.getDataType());
            Expr tensorVarExpr = Var::make(t.getName(), childExpr.getDataType());
            ctx.temporaries.insert({t, tensorVarExpr});
            childTarget.tensor = tensorVarExpr;
            childTarget.pos    = Expr();
            if (emitCompute) {
              Expr zero = ir::Literal::zero(tensorVarExpr.type());
              caseBody.push_back(VarDecl::make(tensorVarExpr, zero));
            }

            // Rewrite lqExpr to substitute the expression computed at the next
            // level with the temporary
            IndexExpr childVar = taco::Access(t);
            lqexpr = replace(lqexpr, {{childExpr,childVar}});
            childVars.push_back(childVar);
          }

          auto childCode = lower(childTarget, child, childExpr, exhausted, ctx);
          util::append(caseBody, childCode);
        }

        // Emit code to compute and store/assign result
        if (emitCompute && (ivarCase==LAST_FREE || ivarCase==BELOW_LAST_FREE)) {
          /// Multiply expressions computed sub-expressions
          auto currentExprs = getAvailableExpressions(lqexpr, iterationGraph.getAncestors(indexVar));
          auto factors = util::combine(currentExprs,childVars);
          taco_iassert(factors.size() > 0);
          IndexExpr expr = factors[0];
          for (auto& factor : util::excludeFirst(factors)) {
            expr = expr * factor;
          }
          emitComputeExpr(target, indexVar, expr, ctx, &caseBody, accumulate);
        }
      }

      if (resultIterator.defined()) {
        Iterator nextResultIterator = (ivarCase == LAST_FREE) ? Iterator() :
            ctx.iterators[resultPath.getStep(resultStep.getStep() + 1)];
        if (!nextResultIterator.defined() ||
            !nextResultIterator.isBranchless()) {
          Expr resultPos = resultIterator.getPosVar();

          std::vector<Stmt> assemblyStmts;

          if (emitAssemble) {
            if (resultIterator.hasAppend()) {
              Stmt appendCoord = resultIterator.getAppendCoord(resultPos, idx);

              if (appendCoord.defined()) {
                assemblyStmts.push_back(appendCoord);
              }
            } else {
              taco_iassert(resultIterator.hasInsert());

              const auto idxVars = getIdxVars(ctx.idxVars, resultIterator, true);
              Stmt insertCoord = resultIterator.getInsertCoord(resultPos, idxVars);

              if (insertCoord.defined()) {
                assemblyStmts.push_back(insertCoord);
              }
            }
          }

          if (resultIterator.hasAppend() && (emitAssemble ||
              ivarCase == LAST_FREE)) {
            Expr nextPos = ir::Add::make(resultPos, 1ll);
            Stmt incPos = Assign::make(resultPos, nextPos);
            assemblyStmts.push_back(incPos);
          }

          Iterator resIter = resultIterator;
          while (resIter.isBranchless()) {
            if (emitAssemble && resIter.hasAppend()) {
              Expr resPos = resIter.getPosVar();
              Expr resParentPos = resIter.getParent().getPosVar();
              Stmt appendEdges = resIter.getAppendEdges(
                  resParentPos, ir::Sub::make(resPos, 1ll), resPos);

              if (appendEdges.defined()) {
                assemblyStmts.push_back(appendEdges);
              }
            }

            resIter = resIter.getParent();
            if (!resIter.getParent().defined()) {
              // No need to emit code for root iterator
              break;
            }

            if (emitAssemble) {
              if (resIter.hasAppend()) {
                Expr resPos = resIter.getPosVar();
                Expr idxVar = ctx.idxVars[resIter];
                Stmt appendCoord = resIter.getAppendCoord(resPos, idxVar);

                if (appendCoord.defined()) {
                  assemblyStmts.push_back(appendCoord);
                }
              //} else {
              //  taco_iassert(resIter.hasInsert());

              //  const auto idxVars = getIdxVars(ctx.idxVars, resIter, true);
              //  Stmt insertCoord = resIter.getInsertCoord(resPos, idxVars);
              //  assemblyStmts.push_back(insertCoord);
              }
            }

            if (resIter.hasAppend()) {
              Expr resPos = resIter.getPosVar();
              Stmt incPos = Assign::make(resPos, ir::Add::make(resPos, 1ll));
              assemblyStmts.push_back(incPos);

              Expr initBegin = ir::Sub::make(resPos, 1ll);
              Stmt initEdges = resIter.getAppendInitEdges(initBegin, resPos);
              if (initEdges.defined()) {
                assemblyStmts.push_back(initEdges);
              }
            }
          }

          if (!assemblyStmts.empty()) {
            Stmt assemblyCode = Block::make(assemblyStmts);
            if (nextResultIterator.defined() &&
                nextResultIterator.hasAppend()) {
              Expr shouldAssemble = Lt::make(nextResultIterator.getBeginVar(),
                                             nextResultIterator.getPosVar());
              assemblyCode = IfThenElse::make(shouldAssemble, assemblyCode);
            }
            caseBody.push_back(assemblyCode);
          }
        }
      }

      // TODO: when merging with switch statement, case bodies need to check
      //       whether inputs accessed with locate are non-zero
      const auto caseIterators = removeIterator(idx, lqRangeIterators);
      Expr cond = mergeWithSwitch ?
          indicatorMask(lpRangeIterators, caseIterators) : [&]() {
          Expr allEqual = allEqualTo(caseIterators, idx);
          Expr allValid = allValidDerefs(lqLocateIterators, guardedIters);
          return simplify(And::make(allEqual, allValid));
        }();
      cases.push_back({cond, Block::make(caseBody)});
    }
    mergeCode.push_back(createIfStatements(cases, lpLattice, ind));

    // Emit code to increment sequential access `pos` variables. Variables that
    // may not be consumed in an iteration (i.e. their iteration space is
    // different from the loop iteration space) are guarded by a conditional:
    // TODO: handle increment of non-unique iterators
    if (emitMerge) {
      // pB1 += (k == kB);
      // pc0 += (k == kc);
      if (mergeWithSwitch) {
        for (size_t i = 0; i < lpRangeIterators.size(); ++i) {
          Iterator iterator = lpRangeIterators[i];
          Expr ivar = iterator.getIteratorVar();
          Expr cmpExpr = Neq::make(BitAnd::make(ind, 1ull << i), 0ull);
          Expr incExpr = Cast::make(cmpExpr, ivar.type());
          Stmt incIVar = Assign::make(ivar, ir::Add::make(ivar, incExpr));
          mergeCode.push_back(incIVar);
        }
      } else {
        for (const auto& iterator : lpRangeIterators) {
          Expr ivar = iterator.getIteratorVar();
          Expr incExpr = (iterator.getCoordVar() == idx || iterator.isFull()) ?
              1ll : [&]() {
                Expr tensorIdx = iterator.getCoordVar();
                return Cast::make(Eq::make(tensorIdx, idx), ivar.type());
              }();
          Stmt inc = Assign::make(ivar, ir::Add::make(ivar, incExpr));
          mergeCode.push_back(inc);
        }
      }
    }

    util::append(loopBody, mergeCode);

    // Emit loop (while loop for merges and for loop for non-merges)
    Stmt mergeLoopBody = Block::make(loopBody);
    Stmt mergeLoop = emitMerge ?
        While::make(noneExhausted(lpRangeIterators), mergeLoopBody) : [&]() {
        Iterator iter = lpRangeIterators[0];
        return For::make(iter.getIteratorVar(), iterFunc.getResults()[0],
                         iterFunc.getResults()[1], 1ll, mergeLoopBody,
                         doParallelize(indexVar, iter.getTensor(), ctx));
      }();
    loops.push_back(mergeLoop);
  }
  util::append(code, loops);

  // Emit a store of the  segment size to the result pos index
  // A2_pos_arr[A1_pos + 1] = A2_pos;
  if (emitAssemble && resultIterator.defined() && resultIterator.hasAppend() &&
      !resultIterator.isBranchless()) {
    Expr resultParentPos = resultIterator.getParent().getPosVar();
    Stmt appendEdges = resultIterator.getAppendEdges(resultParentPos,
        resultIterator.getBeginVar(), resultIterator.getPosVar());

    if (appendEdges.defined()) {
      code.push_back(appendEdges);
    }
  }

  return code;
}

Stmt lower(Assignment assignment, string functionName, set<Property> properties,
           long long allocSize) {
  TensorVar tensorVar = assignment.getLhs().getTensorVar();
  auto name = tensorVar.getName();
  auto indexExpr = assignment.getRhs();
  auto freeVars = assignment.getFreeVars();

  const bool emitAssemble = util::contains(properties, Assemble);
  const bool emitCompute = util::contains(properties, Compute);
  taco_iassert(emitAssemble || emitCompute);

  taco_tassert(!assignment.getOperator().defined() ||
               isa<AddNode>(assignment.getOperator().ptr));
  if (isa<AddNode>(assignment.getOperator().ptr)) {
    properties.insert(Accumulate);
  }

  Schedule schedule = tensorVar.getSchedule();

  // Pack the tensor and it's expression operands into the parameter list
  vector<Expr> parameters;
  vector<Expr> results;
  map<TensorVar,Expr> tensorVars;
  tie(parameters,results,tensorVars) = getTensorVars(assignment);
  taco_iassert(results.size() == 1) << "An expression can only have one result";

  IterationGraph iterationGraph = IterationGraph::make(assignment);
  Ctx ctx(iterationGraph, properties, tensorVars);

  std::vector<Stmt> init, body, finalize;

  // Lower the iteration graph
  auto& roots = ctx.iterationGraph.getRoots();
  TensorPath resultPath = ctx.iterationGraph.getResultTensorPath();

  // Lower tensor expressions
  if (roots.size() > 0) {
    Iterator resultIterator = (resultPath.getSize() > 0)
        ? ctx.iterators[resultPath.getLastStep()]
        : ctx.iterators.getRoot(resultPath);  // e.g. `a = b(i) * c(i)`
    Target target;
    target.tensor = GetProperty::make(resultIterator.getTensor(),
                                      TensorProperty::Values);
    target.pos = resultIterator.getPosVar();

    Expr prevSz = 1ll;
    for (auto& indexVar : resultPath.getVariables()) {
      Iterator iter = ctx.iterators[resultPath.getStep(indexVar)];
      Expr sz = iter.hasAppend() ? 0ll :
                simplify(ir::Mul::make(prevSz, iter.getSize()));

      if (emitAssemble) {
        Stmt initLevel = iter.hasAppend() ?
                         iter.getAppendInitLevel(prevSz, sz) :
                         iter.getInsertInitLevel(prevSz, sz);
        if (initLevel.defined()) {
          init.push_back(initLevel);
        }
      }

      if (iter.hasAppend() && (emitAssemble ||
          indexVar == resultPath.getVariables().back())) {
        // Emit code to initialize result pos variable
        Stmt initIter = VarDecl::make(iter.getPosVar(), 0ll);
        body.push_back(initIter);
      }

      prevSz = sz;
    }

    if (emitCompute) {
      Expr valsSize = GetProperty::make(resultIterator.getTensor(),
                                        TensorProperty::ValuesSize);
      Expr sz = (isa<ir::Literal>(prevSz) &&
                 to<ir::Literal>(prevSz)->equalsScalar(0)) ?
                (emitAssemble ? allocSize : valsSize) : prevSz;

      if (emitAssemble) {
        const std::string valsCapacityName = name + "_vals_capacity";
        ctx.valsCapacity = Var::make(valsCapacityName, Int());

        Stmt initValsCapacity = VarDecl::make(ctx.valsCapacity, sz);
        Stmt allocVals = Allocate::make(target.tensor, sz);

        init.push_back(initValsCapacity);
        init.push_back(allocVals);
      }

      // Emit code to zero result value array, if the output is dense and if
      // either an output mode is merged with a sparse input mode or if the
      // emitted code is a scatter code.
      Expr zero = ir::Literal::zero(target.tensor.type());
      if (!util::contains(properties, Accumulate)) {
        if (resultPath.getSize() == 0) {
          taco_iassert(isa<ir::Literal>(sz)&&
                       to<ir::Literal>(sz)->equalsScalar(1));
          body.push_back(Store::make(target.tensor, 0ll, zero));
        } else if (resultIterator.hasInsert() && needsZero(ctx) &&
                   (!isa<ir::Literal>(sz) ||
                   !to<ir::Literal>(sz)->equalsScalar(allocSize))) {
          Expr iterVar = Var::make("p" + name, Int());
          Stmt zeroStmt = Store::make(target.tensor, iterVar, zero);
          body.push_back(For::make(iterVar, 0ll, sz, 1ll, zeroStmt));
        }
      }
    }

    for (auto& root : roots) {
      // TODO: check if generated loop nest is required (i.e., if it modifies
      //       output arrays)
      auto loopNest = lower(target, root, indexExpr, {}, ctx);
      util::append(body, loopNest);
    }

    if (emitAssemble) {
      Expr prevSz = 1ll;
      for (auto& indexVar : resultPath.getVariables()) {
        Iterator iter = ctx.iterators[resultPath.getStep(indexVar)];
        Expr sz = iter.hasAppend() ? iter.getPosVar() :
                  simplify(ir::Mul::make(prevSz, iter.getSize()));

        Stmt finalizeLevel = iter.hasAppend() ?
                             iter.getAppendFinalizeLevel(prevSz, sz) :
                             iter.getInsertFinalizeLevel(prevSz, sz);
        if (finalizeLevel.defined()) {
          finalize.push_back(finalizeLevel);
        }

        prevSz = sz;
      }

      // Allocate values array after assembling indices if not simultaneously
      // performing compute.
      if (!emitCompute) {
        Expr valsSize = GetProperty::make(resultIterator.getTensor(),
                                          TensorProperty::ValuesSize);

        Stmt allocVals = Allocate::make(target.tensor, prevSz);
        Stmt storeValsSize = Assign::make(valsSize, prevSz);

        finalize.push_back(allocVals);
        finalize.push_back(storeValsSize);
      }
    }
  }
  // Lower scalar expressions
  else {
    TensorPath resultPath = ctx.iterationGraph.getResultTensorPath();
    Expr resultTensorVar = ctx.iterators.getRoot(resultPath).getTensor();
    Expr vals = GetProperty::make(resultTensorVar, TensorProperty::Values);
    if (emitAssemble) {
      Stmt allocVals = Allocate::make(vals, 1ll);
      init.push_back(allocVals);
    }
    if (emitCompute) {
      Expr expr = lowerToScalarExpression(indexExpr, ctx.iterators,
                                          ctx.iterationGraph,
                                          map<TensorVar,Expr>());
      Stmt compute = Store::make(vals, 0ll, expr);
      body.push_back(compute);
    }
  }

  if (!init.empty()) {
    init.push_back(BlankLine::make());
    body = util::combine(init, body);
  }
  if (!finalize.empty()) {
    body.push_back(BlankLine::make());
    util::append(body, finalize);
  }

  return Function::make(functionName, results, parameters, Block::make(body));
}

}}
