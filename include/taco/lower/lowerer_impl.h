#ifndef TACO_LOWERER_IMPL_H
#define TACO_LOWERER_IMPL_H

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

class LowererImpl : public util::Uncopyable {
public:
  LowererImpl();
  virtual ~LowererImpl() = default;

  /// Lower an index statement to an IR function.
  virtual ir::Stmt lower(IndexStmt stmt, std::string name, 
                 bool assemble, bool compute, bool pack, bool unpack) = 0;

protected:

  /// Lower an assignment statement.
  virtual ir::Stmt lowerAssignment(Assignment assignment) = 0;

  /// Lower a yield statement.
  virtual ir::Stmt lowerYield(Yield yield) = 0;


  /// Lower a forall statement.
  virtual ir::Stmt lowerForall(Forall forall) = 0;

  /// Lower a forall that needs to be cloned so that one copy does not have guards
  /// used for vectorized and unrolled loops
  virtual ir::Stmt lowerForallCloned(Forall forall) = 0;

  /// Lower a forall that iterates over all the coordinates in the forall index
  /// var's dimension, and locates tensor positions from the locate iterators.
  virtual ir::Stmt lowerForallDimension(Forall forall,
                                        std::vector<Iterator> locaters,
                                        std::vector<Iterator> inserters,
                                        std::vector<Iterator> appenders,
                                        std::set<Access> reducedAccesses,
                                        ir::Stmt recoveryStmt) = 0;

  /// Lower a forall that iterates over all the coordinates in the forall index
  /// var's dimension, and locates tensor positions from the locate iterators.
  virtual ir::Stmt lowerForallDenseAcceleration(Forall forall,
                                                std::vector<Iterator> locaters,
                                                std::vector<Iterator> inserters,
                                                std::vector<Iterator> appenders,
                                                std::set<Access> reducedAccesses,
                                                ir::Stmt recoveryStmt) = 0;


  /// Lower a forall that iterates over the coordinates in the iterator, and
  /// locates tensor positions from the locate iterators.
  virtual ir::Stmt lowerForallCoordinate(Forall forall, Iterator iterator,
                                         std::vector<Iterator> locaters,
                                         std::vector<Iterator> inserters,
                                         std::vector<Iterator> appenders,
                                         std::set<Access> reducedAccesses,
                                         ir::Stmt recoveryStmt) = 0;

  /// Lower a forall that iterates over the positions in the iterator, accesses
  /// the iterators coordinate, and locates tensor positions from the locate
  /// iterators.
  virtual ir::Stmt lowerForallPosition(Forall forall, Iterator iterator,
                                       std::vector<Iterator> locaters,
                                       std::vector<Iterator> inserters,
                                       std::vector<Iterator> appenders,
                                       std::set<Access> reducedAccesses,
                                       ir::Stmt recoveryStmt) = 0;

  virtual ir::Stmt lowerForallFusedPosition(Forall forall, Iterator iterator,
                                       std::vector<Iterator> locaters,
                                       std::vector<Iterator> inserters,
                                       std::vector<Iterator> appenders,
                                       std::set<Access> reducedAccesses,
                                       ir::Stmt recoveryStmt) = 0;

  /// Used in lowerForallFusedPosition to generate code to
  /// search for the start of the iteration of the loop (a separate kernel on GPUs)
  virtual ir::Stmt searchForFusedPositionStart(Forall forall, Iterator posIterator) = 0;

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
  virtual ir::Stmt lowerMergeLattice(MergeLattice lattice, IndexVar coordinateVar,
                                     IndexStmt statement, 
                                     const std::set<Access>& reducedAccesses) = 0;

  virtual ir::Stmt resolveCoordinate(std::vector<Iterator> mergers, ir::Expr coordinate, bool emitVarDecl) = 0;

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
                                   ir::Expr coordinate, IndexVar coordinateVar, IndexStmt statement,
                                   const std::set<Access>& reducedAccesses, bool resolvedCoordDeclared) = 0;

  /// Lower a merge lattice to cases.
  virtual ir::Stmt lowerMergeCases(ir::Expr coordinate, IndexVar coordinateVar, IndexStmt stmt,
                                   MergeLattice lattice,
                                   const std::set<Access>& reducedAccesses) = 0;

  /// Lower a forall loop body.
  virtual ir::Stmt lowerForallBody(ir::Expr coordinate, IndexStmt stmt,
                                   std::vector<Iterator> locaters,
                                   std::vector<Iterator> inserters,
                                   std::vector<Iterator> appenders,
                                   const std::set<Access>& reducedAccesses) = 0;


  /// Lower a where statement.
  virtual ir::Stmt lowerWhere(Where where) = 0;

  /// Lower a sequence statement.
  virtual ir::Stmt lowerSequence(Sequence sequence) = 0;

  /// Lower an assemble statement.
  virtual ir::Stmt lowerAssemble(Assemble assemble) = 0;

  /// Lower a multi statement.
  virtual ir::Stmt lowerMulti(Multi multi) = 0;

  /// Lower a suchthat statement.
  virtual ir::Stmt lowerSuchThat(SuchThat suchThat) = 0;

  /// Lower an access expression.
  virtual ir::Expr lowerAccess(Access access) = 0;

  /// Lower a literal expression.
  virtual ir::Expr lowerLiteral(Literal literal) = 0;

  /// Lower a negate expression.
  virtual ir::Expr lowerNeg(Neg neg) = 0;
	
  /// Lower an addition expression.
  virtual ir::Expr lowerAdd(Add add) = 0;

  /// Lower a subtraction expression.
  virtual ir::Expr lowerSub(Sub sub) = 0;

  /// Lower a multiplication expression.
  virtual ir::Expr lowerMul(Mul mul) = 0;

  /// Lower a division expression.
  virtual ir::Expr lowerDiv(Div div) = 0;

  /// Lower a square root expression.
  virtual ir::Expr lowerSqrt(Sqrt sqrt) = 0;

  /// Lower a cast expression.
  virtual ir::Expr lowerCast(Cast cast) = 0;

  /// Lower an intrinsic function call expression.
  virtual ir::Expr lowerCallIntrinsic(CallIntrinsic call) = 0;


  /// Lower a concrete index variable statement.
  virtual ir::Stmt lower(IndexStmt stmt);

  /// Lower a concrete index variable expression.
  virtual ir::Expr lower(IndexExpr expr);

  class Visitor;
  friend class Visitor;
  std::shared_ptr<Visitor> visitor;

};

}
#endif
