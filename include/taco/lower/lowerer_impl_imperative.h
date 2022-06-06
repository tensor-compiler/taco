#ifndef TACO_LOWERER_IMPL_IMPERATIVE_H
#define TACO_LOWERER_IMPL_IMPERATIVE_H

#include <utility>
#include <vector>
#include <map>
#include <set>
#include <memory>
#include <taco/index_notation/index_notation.h>

#include "taco/lower/iterator.h"
#include "taco/util/scopedset.h"
#include "taco/util/uncopyable.h"
#include "taco/ir_tags.h"
#include "taco/lower/lowerer_impl.h"

namespace taco {

class TensorVar;
class IndexVar;

class IndexStmt;
class Assignment;
class Yield;
class Forall;
class Where;
class Multi;
class SuchThat;
class Sequence;

class IndexExpr;
class Access;
class Literal;
class Neg;
class Add;
class Sub;
class Mul;
class Div;
class Sqrt;
class Cast;
class CallIntrinsic;

class MergeLattice;
class MergePoint;
class ModeAccess;

namespace ir {
class Stmt;
class Expr;
}

class LowererImplImperative : public LowererImpl {
public:
  LowererImplImperative();
  virtual ~LowererImplImperative() = default;

  /// Lower an index statement to an IR function.
  ir::Stmt lower(IndexStmt stmt, std::string name, 
                 bool assemble, bool compute, bool pack, bool unpack);

protected:

  /// Lower an assignment statement.
  virtual ir::Stmt lowerAssignment(Assignment assignment);

  /// Lower a yield statement.
  virtual ir::Stmt lowerYield(Yield yield);


  /// Lower a forall statement.
  virtual ir::Stmt lowerForall(Forall forall);

  /// Lower a forall that needs to be cloned so that one copy does not have guards
  /// used for vectorized and unrolled loops
  virtual ir::Stmt lowerForallCloned(Forall forall);

  /// Lower a forall that iterates over all the coordinates in the forall index
  /// var's dimension, and locates tensor positions from the locate iterators.
  virtual ir::Stmt lowerForallDimension(Forall forall,
                                        std::vector<Iterator> locaters,
                                        std::vector<Iterator> inserters,
                                        std::vector<Iterator> appenders,
                                        MergeLattice caseLattice,
                                        std::set<Access> reducedAccesses,
                                        ir::Stmt recoveryStmt);

  /// Lower a forall that iterates over all the coordinates in the forall index
  /// var's dimension, and locates tensor positions from the locate iterators.
  virtual ir::Stmt lowerForallDenseAcceleration(Forall forall,
                                                std::vector<Iterator> locaters,
                                                std::vector<Iterator> inserters,
                                                std::vector<Iterator> appenders,
                                                MergeLattice caseLattice,
                                                std::set<Access> reducedAccesses,
                                                ir::Stmt recoveryStmt);


  /// Lower a forall that iterates over the coordinates in the iterator, and
  /// locates tensor positions from the locate iterators.
  virtual ir::Stmt lowerForallCoordinate(Forall forall, Iterator iterator,
                                         std::vector<Iterator> locaters,
                                         std::vector<Iterator> inserters,
                                         std::vector<Iterator> appenders,
                                         MergeLattice caseLattice,
                                         std::set<Access> reducedAccesses,
                                         ir::Stmt recoveryStmt);

  /// Lower a forall that iterates over the positions in the iterator, accesses
  /// the iterators coordinate, and locates tensor positions from the locate
  /// iterators.
  virtual ir::Stmt lowerForallPosition(Forall forall, Iterator iterator,
                                       std::vector<Iterator> locaters,
                                       std::vector<Iterator> inserters,
                                       std::vector<Iterator> appenders,
                                       MergeLattice caseLattice,
                                       std::set<Access> reducedAccesses,
                                       ir::Stmt recoveryStmt);

  virtual ir::Stmt lowerForallFusedPosition(Forall forall, Iterator iterator,
                                       std::vector<Iterator> locaters,
                                       std::vector<Iterator> inserters,
                                       std::vector<Iterator> appenders,
                                       MergeLattice caseLattice,
                                       std::set<Access> reducedAccesses,
                                       ir::Stmt recoveryStmt);

  /// Used in lowerForallFusedPosition to generate code to
  /// search for the start of the iteration of the loop (a separate kernel on GPUs)
  virtual ir::Stmt searchForFusedPositionStart(Forall forall, Iterator posIterator);

    /**
     * Lower the merge lattice to code that iterates over the sparse iteration
     * space of coordinates and computes the concrete index notation statement.
     * The merge lattice dictates the code to iterate over the coordinates, by
     * successively iterating to the exhaustion of each relevant sparse iteration
     * space region (i.e., the regions in a venn diagram).  The statement is then
     * computed and/or indices assembled at each point in its sparse iteration
     * space.
     *
     * \param lattice
     *      A merge lattice that describes the sparse iteration space of the
     *      concrete index notation statement.
     * \param coordinate
     *      An IR expression that resolves to the variable containing the current
     *      coordinate the merge lattice is at.
     * \param statement
     *      A concrete index notation statement to compute at the points in the
     *      sparse iteration space described by the merge lattice.
     * \param mergeStrategy
     *      A strategy for merging iterators. One of TwoFinger or Gallop.
     *
     * \return
     *       IR code to compute the forall loop.
     */
  virtual ir::Stmt lowerMergeLattice(MergeLattice lattice, IndexVar coordinateVar,
                                     IndexStmt statement, 
                                     const std::set<Access>& reducedAccesses, 
                                     MergeStrategy mergeStrategy);

  virtual ir::Stmt resolveCoordinate(std::vector<Iterator> mergers, ir::Expr coordinate, bool emitVarDecl, bool mergeWithMax);

    /**
     * Lower the merge point at the top of the given lattice to code that iterates
     * until one region of the sparse iteration space of coordinates and computes
     * the concrete index notation statement.
     *
     * \param pointLattice
     *      A merge lattice whose top point describes a region of the sparse
     *      iteration space of the concrete index notation statement.
     * \param coordinate
     *      An IR expression that resolves to the variable containing the current
     *      coordinate the merge point is at.
     *      A concrete index notation statement to compute at the points in the
     *      sparse iteration space region described by the merge point.
     * \param mergeWithMax
     *      A boolean indicating whether coordinates should be combined with MAX instead of MIN.
     *      MAX is needed when the iterators are merged with the Gallop strategy.
     */
  virtual ir::Stmt lowerMergePoint(MergeLattice pointLattice,
                                   ir::Expr coordinate, IndexVar coordinateVar, IndexStmt statement,
                                   const std::set<Access>& reducedAccesses, bool resolvedCoordDeclared, 
                                   MergeStrategy mergestrategy);

  /// Lower a merge lattice to cases.
  virtual ir::Stmt lowerMergeCases(ir::Expr coordinate, IndexVar coordinateVar, IndexStmt stmt,
                                   MergeLattice lattice,
                                   const std::set<Access>& reducedAccesses, 
                                   MergeStrategy mergeStrategy);

  /// Lower a forall loop body.
  virtual ir::Stmt lowerForallBody(ir::Expr coordinate, IndexStmt stmt,
                                   std::vector<Iterator> locaters,
                                   std::vector<Iterator> inserters,
                                   std::vector<Iterator> appenders,
                                   MergeLattice caseLattice,
                                   const std::set<Access>& reducedAccesses, 
                                   MergeStrategy mergeStrategy);


  /// Lower a where statement.
  virtual ir::Stmt lowerWhere(Where where);

  /// Lower a sequence statement.
  virtual ir::Stmt lowerSequence(Sequence sequence);

  /// Lower an assemble statement.
  virtual ir::Stmt lowerAssemble(Assemble assemble);

  /// Lower a multi statement.
  virtual ir::Stmt lowerMulti(Multi multi);

  /// Lower a suchthat statement.
  virtual ir::Stmt lowerSuchThat(SuchThat suchThat);

  /// Lower an access expression.
  virtual ir::Expr lowerAccess(Access access);

  /// Lower a literal expression.
  virtual ir::Expr lowerLiteral(Literal literal);

  /// Lower a negate expression.
  virtual ir::Expr lowerNeg(Neg neg);
	
  /// Lower an addition expression.
  virtual ir::Expr lowerAdd(Add add);

  /// Lower a subtraction expression.
  virtual ir::Expr lowerSub(Sub sub);

  /// Lower a multiplication expression.
  virtual ir::Expr lowerMul(Mul mul);

  /// Lower a division expression.
  virtual ir::Expr lowerDiv(Div div);

  /// Lower a square root expression.
  virtual ir::Expr lowerSqrt(Sqrt sqrt);

  /// Lower a cast expression.
  virtual ir::Expr lowerCast(Cast cast);

  /// Lower an intrinsic function call expression.
  virtual ir::Expr lowerCallIntrinsic(CallIntrinsic call);

  /// Lower an IndexVar expression
  virtual ir::Expr lowerIndexVar(IndexVar var);

  /// Lower a generic tensor operation expression
  virtual ir::Expr lowerTensorOp(Call op);

  /// Lower a concrete index variable statement.
  ir::Stmt lower(IndexStmt stmt);

  /// Lower a concrete index variable expression.
  ir::Expr lower(IndexExpr expr);

  /// Check whether the lowerer should generate code to assemble result indices.
  bool generateAssembleCode() const;

  /// Check whether the lowerer should generate code to compute result values.
  bool generateComputeCode() const;


  /// Retrieve a tensor IR variable.
  ir::Expr getTensorVar(TensorVar) const;

  /// Retrieves a result values array capacity variable.
  ir::Expr getCapacityVar(ir::Expr) const;

  /// Retrieve the values array of the tensor var.
  ir::Expr getValuesArray(TensorVar) const;

  /// Retrieve the dimension of an index variable (the values it iterates over),
  /// which is encoded as the interval [0, result).
  ir::Expr getDimension(IndexVar indexVar) const;

  /// Retrieve the chain of iterators that iterate over the access expression.
  std::vector<Iterator> getIterators(Access) const;

  /// Retrieve the access expressions that have been exhausted.
  std::set<Access> getExhaustedAccesses(MergePoint, MergeLattice) const;

  /// Retrieve the reduced tensor component value corresponding to an access.
  ir::Expr getReducedValueVar(Access) const;

  /// Retrieve the coordinate IR variable corresponding to an index variable.
  ir::Expr getCoordinateVar(IndexVar) const;

  /// Retrieve the coordinate IR variable corresponding to an iterator.
  ir::Expr getCoordinateVar(Iterator) const;


  /**
   * Retrieve the resolved coordinate variables of an iterator and it's parent
   * iterators, which are the coordinates after per-iterator coordinates have
   * been merged with the min function.
   *
   * \param iterator
   *      A defined iterator (that take part in a chain of parent iterators).
   *
   * \return
   *       IR expressions that resolve to resolved coordinates for the
   *       iterators.  The first entry is the resolved coordinate of this
   *       iterator followed by its parent's, its grandparent's, etc.
   */
  std::vector<ir::Expr> coordinates(Iterator iterator) const;

  /**
   * Retrieve the resolved coordinate variables of the iterators, which are the
   * coordinates after per-iterator coordinates have been merged with the min
   * function.
   *
   * \param iterators
   *      A set of defined iterators.
   *
   * \return
   *      IR expressions that resolve to resolved coordinates for the iterators,
   *      in the same order they were given.
   */
  std::vector<ir::Expr> coordinates(std::vector<Iterator> iterators);

  /// Generate code to initialize result indices.
  ir::Stmt initResultArrays(std::vector<Access> writes, 
                            std::set<Access> reducedAccesses);

  /// Generate code to finalize result indices.
  ir::Stmt finalizeResultArrays(std::vector<Access> writes);

  /**
   * Replace scalar tensor pointers with stack scalar for lowering.
   */
  ir::Stmt defineScalarVariable(TensorVar var, bool zero);

  ir::Stmt initResultArrays(IndexVar var, std::vector<Access> writes,
                            std::set<Access> reducedAccesses);

  ir::Stmt resizeAndInitValues(const std::vector<Iterator>& appenders,
                               const std::set<Access>& reducedAccesses);
  /**
   * Generate code to initialize values array in range
   * [begin * size, (begin + 1) * size) with the fill value.
   */
  ir::Stmt initValues(ir::Expr tensor, ir::Expr initVal, ir::Expr begin, ir::Expr size);

  /// Declare position variables and initialize them with a locate.
  ir::Stmt declLocatePosVars(std::vector<Iterator> iterators);

  /// Emit loops to reduce duplicate coordinates.
  ir::Stmt reduceDuplicateCoordinates(ir::Expr coordinate, 
                                      std::vector<Iterator> iterators, 
                                      bool alwaysReduce);

  /**
   * Create code to declare and initialize while loop iteration variables,
   * including both pos variables (of e.g. compressed modes) and crd variables
   * (e.g. dense modes).
   *
   * \param iterators
   *      Iterators whose iteration variables will be declared and initialized.
   *
   * \return
   *      A IR statement that declares and initializes each iterator's iterators
   *      variable
   */
  ir::Stmt codeToInitializeIteratorVars(std::vector<Iterator> iterators, std::vector<Iterator> rangers, std::vector<Iterator> mergers, ir::Expr coord, IndexVar coordinateVar);
  ir::Stmt codeToInitializeIteratorVar(Iterator iterator, std::vector<Iterator> iterators, std::vector<Iterator> rangers, std::vector<Iterator> mergers, ir::Expr coordinate, IndexVar coordinateVar);

  /// Returns true iff the temporary used in the where statement is dense and sparse iteration over that
  /// temporary can be automaticallty supported by the compiler.
  std::pair<bool,bool> canAccelerateDenseTemp(Where where);

  /// Initializes a temporary workspace
  std::vector<ir::Stmt> codeToInitializeTemporary(Where where);
  std::vector<ir::Stmt> codeToInitializeTemporaryParallel(Where where, ParallelUnit parallelUnit);
  std::vector<ir::Stmt> codeToInitializeLocalTemporaryParallel(Where where, ParallelUnit parallelUnit);
  /// Gets the size of a temporary tensorVar in the where statement
  ir::Expr getTemporarySize(Where where);

  /// Initializes helper arrays to give dense workspaces sparse acceleration
  std::vector<ir::Stmt> codeToInitializeDenseAcceleratorArrays(Where where, bool parallel = false);

  /// Recovers a derived indexvar from an underived variable.
  ir::Stmt codeToRecoverDerivedIndexVar(IndexVar underived, IndexVar indexVar, bool emitVarDecl);

  /// Conditionally increment iterator position variables.
  ir::Stmt codeToIncIteratorVars(ir::Expr coordinate, IndexVar coordinateVar,
          std::vector<Iterator> iterators, std::vector<Iterator> mergers, MergeStrategy strategy);

  ir::Stmt codeToLoadCoordinatesFromPosIterators(std::vector<Iterator> iterators, bool declVars);

  /// Create statements to append coordinate to result modes.
  ir::Stmt appendCoordinate(std::vector<Iterator> appenders, ir::Expr coord);

  /// Create statements to append positions to result modes.
  ir::Stmt generateAppendPositions(std::vector<Iterator> appenders);

  /// Create an expression to index into a tensor value array.
  ir::Expr generateValueLocExpr(Access access) const;

  /// Expression that evaluates to true if none of the iterators are exhausted
  ir::Expr checkThatNoneAreExhausted(std::vector<Iterator> iterators);

  /// Create an expression that can be used to filter out (some) zeros in the
  /// result
  ir::Expr generateAssembleGuard(IndexExpr expr);

  /// Check whether the result tensor should be assembled by ungrouped insertion
  bool isAssembledByUngroupedInsertion(TensorVar result);
  bool isAssembledByUngroupedInsertion(ir::Expr result);

  bool isNonFullyInitialized(ir::Expr result);

  /// Check whether the statement writes to a result tensor
  bool hasStores(ir::Stmt stmt);

  std::pair<std::vector<Iterator>,std::vector<Iterator>>
  splitAppenderAndInserters(const std::vector<Iterator>& results);

  /// Lowers a merge lattice to cases assuming there are no more loops to be emitted in stmt.
  /// Will emit checks for explicit zeros for each mode iterator and each locator in the lattice.
  ir::Stmt lowerMergeCasesWithExplicitZeroChecks(ir::Expr coordinate, IndexVar coordinateVar, IndexStmt stmt,
                                                 MergeLattice lattice, const std::set<Access>& reducedAccesses, 
                                                 MergeStrategy mergeStrategy);

  /// Constructs cases comparing the coordVar for each iterator to the resolved coordinate.
  /// Returns a vector where coordComparisons[i] corresponds to a case for iters[i]
  /// If no case can be formed for a given iterator, an undefined expr is appended where a case would normally be.
  template<typename C>
  std::vector<ir::Expr> compareToResolvedCoordinate(const std::vector<Iterator>& iters, ir::Expr resolvedCoordinate,
                                                    IndexVar coordinateVar) {
    std::vector<ir::Expr> coordComparisons;

    for (Iterator iterator : iters) {
      if (!(provGraph.isCoordVariable(iterator.getIndexVar()) &&
            provGraph.isDerivedFrom(iterator.getIndexVar(), coordinateVar))) {
        coordComparisons.push_back(C::make(iterator.getCoordVar(), resolvedCoordinate));
      } else {
        coordComparisons.push_back(ir::Expr());
      }
    }

    return coordComparisons;
  }

  /// Makes the preamble of booleans used in case checks for the inner most loop of the computations
  /// The iterator to condition map contains the name of the boolean indicating if each corresponding mode iterator
  /// and each locator is non-zero. This function populates this map so the caller can user the boolean names to emit
  /// checks for each lattice point.
  std::vector<ir::Stmt> constructInnerLoopCasePreamble(ir::Expr coordinate, IndexVar coordinateVar,
                                                       MergeLattice lattice,
                                                       std::map<Iterator, ir::Expr>& iteratorToConditionMap);

  /// Lowers merge cases in the lattice using a map to know what expr to emit for each iterator in the lattice.
  /// The map must be of iterators to exprs of boolean types
  std::vector<ir::Stmt> lowerCasesFromMap(std::map<Iterator, ir::Expr> iteratorToCondition,
                                          ir::Expr coordinate, IndexStmt stmt, const MergeLattice& lattice,
                                          const std::set<Access>& reducedAccesses, MergeStrategy mergeStrategy);

  /// Constructs an expression which checks if this access is "zero"
  ir::Expr constructCheckForAccessZero(Access);

  /// Filters out a list of iterators and returns those the lowerer should explicitly check for zeros.
  /// For now, we only check mode iterators.
  std::vector<Iterator> getModeIterators(const std::vector<Iterator>&);

  /// Emit early exit
  ir::Stmt emitEarlyExit(ir::Expr reductionExpr, std::vector<Property>&);

  /// Expression that returns the beginning of a window to iterate over
  /// in a compressed iterator. It is used when operating over windows of
  /// tensors, instead of the full tensor.
  ir::Expr searchForStartOfWindowPosition(Iterator iterator, ir::Expr start, ir::Expr end);

  /// Expression that returns the end of a window to iterate over
  /// in a compressed iterator. It is used when operating over windows of
  /// tensors, instead of the full tensor.
  ir::Expr searchForEndOfWindowPosition(Iterator iterator, ir::Expr start, ir::Expr end);

  /// Statement that guards against going out of bounds of the window that
  /// the input iterator was configured with.
  ir::Stmt upperBoundGuardForWindowPosition(Iterator iterator, ir::Expr access);

  /// Expression that recovers a canonical index variable from a position in
  /// a windowed position iterator. A windowed position iterator iterates over
  /// values in the range [lo, hi). This expression projects values in that
  /// range back into the canonical range of [0, n).
  ir::Expr projectWindowedPositionToCanonicalSpace(Iterator iterator, ir::Expr expr);

  // projectCanonicalSpaceToWindowedPosition is the opposite of
  // projectWindowedPositionToCanonicalSpace. It takes an expression ranging
  // through the canonical space of [0, n) and projects it up to the windowed
  // range of [lo, hi).
  ir::Expr projectCanonicalSpaceToWindowedPosition(Iterator iterator, ir::Expr expr);

  /// strideBoundsGuard inserts a guard against accessing values from an
  /// iterator that don't fit in the stride that the iterator is configured
  /// with. It takes a boolean incrementPosVars to control whether the outer
  /// loop iterator variable should be incremented when the guard is fired.
  ir::Stmt strideBoundsGuard(Iterator iterator, ir::Expr access, bool incrementPosVar);

private:
  bool assemble;
  bool compute;
  bool loopOrderAllowsShortCircuit = false;

  std::set<TensorVar> needCompute;

  int markAssignsAtomicDepth = 0;
  ParallelUnit atomicParallelUnit;

  std::set<TensorVar> assembledByUngroupedInsert;

  std::set<ir::Expr> nonFullyInitializedResults;

  /// Map used to hoist temporary workspace initialization
  std::map<Forall, Where> temporaryInitialization;

  /// Map used to hoist parallel temporary workspaces. Maps workspace shared by all threads to where statement
  std::map<Where, TensorVar> whereToTemporaryVar;
  std::map<Where, ir::Expr> whereToIndexListAll;
  std::map<Where, ir::Expr> whereToIndexListSizeAll;
  std::map<Where, ir::Expr> whereToBitGuardAll;

  /// Map from tensor variables in index notation to variables in the IR
  std::map<TensorVar, ir::Expr> tensorVars;

  struct TemporaryArrays {
    ir::Expr values;
  };
  std::map<TensorVar, TemporaryArrays> temporaryArrays;

  /// Map form temporary to indexList var if accelerating dense workspace
  std::map<TensorVar, ir::Expr> tempToIndexList;

  /// Map form temporary to indexListSize if accelerating dense workspace
  std::map<TensorVar, ir::Expr> tempToIndexListSize;

  /// Map form temporary to bitGuard var if accelerating dense workspace
  std::map<TensorVar, ir::Expr> tempToBitGuard;

  std::set<TensorVar> guardedTemps;

  /// Map from result tensors to variables tracking values array capacity.
  std::map<ir::Expr, ir::Expr> capacityVars;

  /// Map from index variables to their dimensions, currently [0, expr).
  std::map<IndexVar, ir::Expr> dimensions;

  /// Map from index variables to their bounds, currently also [0, expr) but allows adding minimum in future too
  std::map<IndexVar, std::vector<ir::Expr>> underivedBounds;

  /// Map from indexvars to their variable names
  std::map<IndexVar, ir::Expr> indexVarToExprMap;

  /// Tensor and mode iterators to iterate over in the lowered code
  Iterators iterators;

  /// Keep track of relations between IndexVars
  ProvenanceGraph provGraph;

  bool ignoreVectorize = false; // already being taken into account

  std::vector<ir::Stmt> whereConsumers;
  std::vector<TensorVar> whereTemps;
  std::map<TensorVar, const AccessNode *> whereTempsToResult;

  // Map temporary tensorVars to a list of size expressions for each mode
  std::map<TensorVar, std::vector<ir::Expr>> temporarySizeMap;
  
  // List that contains all temporary tensorVars
  std::vector<TensorVar> temporaries;

  bool captureNextLocatePos = false;
  ir::Stmt capturedLocatePos; // used for whereConsumer when want to replicate same locating

  bool emitUnderivedGuards = true;

  int inParallelLoopDepth = 0;

  std::map<ParallelUnit, ir::Expr> parallelUnitSizes;
  std::map<ParallelUnit, IndexVar> parallelUnitIndexVars;

  /// Keep track of what IndexVars have already been defined
  std::set<IndexVar> definedIndexVars;
  std::vector<IndexVar> definedIndexVarsOrdered;

  /// Map from tensor accesses to variables storing reduced values.
  std::map<Access, ir::Expr> reducedValueVars;

  /// Set of locate-capable iterators that can be legally accessed.
  util::ScopedSet<Iterator> accessibleIterators;

  /// Visitor methods can add code to emit it to the function header.
  std::vector<ir::Stmt> header;

  /// Visitor methods can add code to emit it to the function footer.
  std::vector<ir::Stmt> footer;

  class Visitor;
  friend class Visitor;
  std::shared_ptr<Visitor> visitor;

};

}
#endif
