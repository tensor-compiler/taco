#ifndef TACO_LOWERER_IMPL_H
#define TACO_LOWERER_IMPL_H

#include <vector>
#include <map>
#include <set>
#include <memory>

#include "taco/lower/iterator.h"
#include "taco/util/scopedset.h"
#include "taco/util/uncopyable.h"

namespace taco {

class TensorVar;
class IndexVar;

class IndexStmt;
class Assignment;
class Yield;
class Forall;
class Where;
class Multi;
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

class LowererImpl : public util::Uncopyable {
public:
  LowererImpl();
  virtual ~LowererImpl() = default;

  /// Lower an index statement to an IR function.
  ir::Stmt lower(IndexStmt stmt, std::string name,  bool assemble, bool compute);

protected:

  /// Lower an assignment statement.
  virtual ir::Stmt lowerAssignment(Assignment assignment);

  /// Lower a yield statement.
  virtual ir::Stmt lowerYield(Yield yield);


  /// Lower a forall statement.
  virtual ir::Stmt lowerForall(Forall forall);

  /// Lower a forall that iterates over all the coordinates in the forall index
  /// var's dimension, and locates tensor positions from the locate iterators.
  virtual ir::Stmt lowerForallDimension(Forall forall,
                                        std::vector<Iterator> locaters,
                                        std::vector<Iterator> inserters,
                                        std::vector<Iterator> appenders,
                                        std::set<Access> reducedAccesses);

  /// Lower a forall that iterates over the coordinates in the iterator, and
  /// locates tensor positions from the locate iterators.
  virtual ir::Stmt lowerForallCoordinate(Forall forall, Iterator iterator,
                                         std::vector<Iterator> locaters,
                                         std::vector<Iterator> inserters,
                                         std::vector<Iterator> appenders,
                                         std::set<Access> reducedAccesses);

  /// Lower a forall that iterates over the positions in the iterator, accesses
  /// the iterators coordinate, and locates tensor positions from the locate
  /// iterators.
  virtual ir::Stmt lowerForallPosition(Forall forall, Iterator iterator,
                                       std::vector<Iterator> locaters,
                                       std::vector<Iterator> inserters,
                                       std::vector<Iterator> appenders,
                                       std::set<Access> reducedAccesses);

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
   *
   * \return
   *       IR code to compute the forall loop.
   */
  virtual ir::Stmt lowerMergeLattice(MergeLattice lattice, ir::Expr coordinate,
                                     IndexStmt statement, 
                                     const std::set<Access>& reducedAccesses);

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
   */
  virtual ir::Stmt lowerMergePoint(MergeLattice pointLattice,
                                   ir::Expr coordinate, IndexStmt statement,
                                   const std::set<Access>& reducedAccesses);

  /// Lower a merge lattice to cases.
  virtual ir::Stmt lowerMergeCases(ir::Expr coordinate, IndexStmt stmt,
                                   MergeLattice lattice,
                                   const std::set<Access>& reducedAccesses);

  /// Lower a forall loop body.
  virtual ir::Stmt lowerForallBody(ir::Expr coordinate, IndexStmt stmt,
                                   std::vector<Iterator> locaters,
                                   std::vector<Iterator> inserters,
                                   std::vector<Iterator> appenders,
                                   const std::set<Access>& reducedAccesses);


  /// Lower a where statement.
  virtual ir::Stmt lowerWhere(Where where);

  /// Lower a sequence statement.
  virtual ir::Stmt lowerSequence(Sequence sequence);

  /// Lower a multi statement.
  virtual ir::Stmt lowerMulti(Multi multi);


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
                            std::vector<Access> reads,
                            std::set<Access> reducedAccesses);

  /// Generate code to finalize result indices.
  ir::Stmt finalizeResultArrays(std::vector<Access> writes);

  /**
   * Replace scalar tensor pointers with stack scalar for lowering.
   */
  ir::Stmt defineScalarVariable(TensorVar var, bool zero);

  ir::Stmt initResultArrays(IndexVar var, std::vector<Access> writes,
                            std::vector<Access> reads,
                            std::set<Access> reducedAccesses);

  ir::Stmt resizeAndInitValues(const std::vector<Iterator>& appenders,
                               const std::set<Access>& reducedAccesses);
  /**
   * Generate code to zero-initialize values array in range
   * [begin * size, (begin + 1) * size).
   */
  ir::Stmt zeroInitValues(ir::Expr tensor, ir::Expr begin, ir::Expr size);

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
  ir::Stmt codeToInitializeIteratorVars(std::vector<Iterator> iterators);

  /// Conditionally increment iterator position variables.
  ir::Stmt codeToIncIteratorVars(ir::Expr coordinate,
                                 std::vector<Iterator> iterators);

  /// Create statements to append coordinate to result modes.
  ir::Stmt appendCoordinate(std::vector<Iterator> appenders, ir::Expr coord);

  /// Create statements to append positions to result modes.
  ir::Stmt generateAppendPositions(std::vector<Iterator> appenders);


  /// Create an expression to index into a tensor value array.
  ir::Expr generateValueLocExpr(Access access) const;

  /// Expression that evaluates to true if none of the iteratators are exhausted
  ir::Expr checkThatNoneAreExhausted(std::vector<Iterator> iterators);

private:
  bool assemble;
  bool compute;

  /// Map from tensor variables in index notation to variables in the IR
  std::map<TensorVar, ir::Expr> tensorVars;

  struct TemporaryArrays {
    ir::Expr values;
  };
  std::map<TensorVar, TemporaryArrays> temporaryArrays;

  /// Map from result tensors to variables tracking values array capacity.
  std::map<ir::Expr, ir::Expr> capacityVars;

  /// Map from index variables to their dimensions, currently [0, expr).
  std::map<IndexVar, ir::Expr> dimensions;

  /// Tensor and mode iterators to iterate over in the lowered code
  Iterators iterators;

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
