#ifndef TACO_LOWERER_IMPL_H
#define TACO_LOWERER_IMPL_H

#include <vector>
#include <map>
#include <memory>
#include "taco/util/uncopyable.h"

namespace taco {

class TensorVar;
class IndexVar;

class IndexStmt;
class Assignment;
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

class Iterator;
class MergeLattice;
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


    /// Lower a forall statement.
  virtual ir::Stmt lowerForall(Forall forall);

  /// Lower a forall that iterates over all the coordinates in the forall index
  /// var's dimension, and locates tensor positions from the locate iterators.
  virtual ir::Stmt lowerForallDimension(Forall forall,
                                        std::vector<Iterator> locateIterators,
                                        std::vector<Iterator> insertIterators,
                                        std::vector<Iterator> appendIterators);

  /// Lower a forall that iterates over the coordinates in the iterator, and
  /// locates tensor positions from the locate iterators.
  virtual ir::Stmt lowerForallCoordinate(Forall forall, Iterator iterator,
                                         std::vector<Iterator> locateIterators,
                                         std::vector<Iterator> insertIterators,
                                         std::vector<Iterator> appendIterators);

  /// Lower a forall that iterates over the positions in the iterator, accesses
  /// the iterators coordinate, and locates tensor positions from the locate
  /// iterators.
  virtual ir::Stmt lowerForallPosition(Forall forall, Iterator iterator,
                                       std::vector<Iterator> locateIterators,
                                       std::vector<Iterator> insertIterators,
                                       std::vector<Iterator> appendIterators);

  /// Lower a forall that merges multiple iterators.
  virtual ir::Stmt lowerForallMerge(Forall forall, MergeLattice lattice);


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

  /// Retrieve the dimension of an index variable (the values it iterates over),
  /// which is encoded as the interval [0, result).
  ir::Expr getDimension(IndexVar) const;

  /// Retrieve the iterator of the mode access.  A mode access is
  /// the access into a tensor by one index variable, for example, the first
  /// mode indexed into by `i` in `B(i,j)`.
  Iterator getIterator(ModeAccess) const;

  /// Retrieve the chain of iterators of the access expression.
  std::vector<Iterator> getIterators(Access) const;

  /// Retrieve a map of one iterator for each mode access.
  const std::map<ModeAccess, Iterator>& getIteratorMap() const;

  /// Retrieve the coordinate IR variable corresponding to an index variable.
  ir::Expr getCoordinateVar(IndexVar) const;

  /// Retrieve the coordinate IR variable corresponding to an iterator.
  ir::Expr getCoordinateVar(Iterator) const;

  /// Retrieve the coordinate variables of iterator and its parents.
  std::vector<ir::Expr> getCoords(Iterator iterator);


  /// Generate code to initialize result indices.
  ir::Stmt generateResultModeInits(std::vector<Access> writes);

  /// Creates code to declare temporaries.
  ir::Stmt generateTemporaryDecls(std::vector<TensorVar> temporaries,
                                  std::map<TensorVar,ir::Expr> scalars);

  /// Create position variable locate declarations for each locate iterator
  ir::Stmt generatePosVarLocateDecls(std::vector<Iterator> locateIterators);

  /// Create an expression to index into a tensor value array.
  ir::Expr generateValueLocExpr(Access access) const;

  /// This function generates the loop body.  This includes calculating located
  /// position variables, appends, inserts, and the recursive call to generate
  /// the sub-statement.
  ir::Stmt generateLoopBody(Forall forall,
                            std::vector<Iterator> locateIterators,
                            std::vector<Iterator> insertIterators,
                            std::vector<Iterator> appendIterators);

private:
  bool assemble;
  bool compute;

  /// Map from tensor variables in index notation to variables in the IR
  std::map<TensorVar, ir::Expr> tensorVars;

  /// Map from index variables to their dimensions, currently [0, expr).
  std::map<IndexVar, ir::Expr> dimensions;

  /// Map from mode accesses to iterators.
  std::map<ModeAccess, Iterator> iterators;

  /// Map from iterators to the index variables they contribute to.
  std::map<Iterator, IndexVar> indexVars;

  /// Map from index variables to corresponding resolved coordinate variable.
  std::map<IndexVar, ir::Expr> coordVars;

  class Visitor;
  friend class Visitor;
  std::shared_ptr<Visitor> visitor;

};

}
#endif
