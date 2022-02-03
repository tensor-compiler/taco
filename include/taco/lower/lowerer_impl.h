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
class Call;
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

  /// Lower a call expression.
  virtual ir::Expr lowerTensorOp(Call call) = 0;

  /// Lower an index variable
  virtual ir::Expr lowerIndexVar(IndexVar indexVar) = 0;

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
