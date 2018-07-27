#include "taco/lower/lowerer_impl.h"

#include "taco/index_notation/index_notation.h"
#include "taco/ir/ir.h"
#include "iterator.h"
#include "merge_lattice.h"
#include "mode_access.h"
#include "taco/util/collections.h"

using namespace std;
using namespace taco::ir;

namespace taco {

Stmt LowererImpl::lower(IndexStmt stmt, string name,
                        bool assemble, bool compute) {
  return Stmt();
}

Stmt LowererImpl::lowerAssignment(Assignment assignment) {
  return Stmt();
}

Stmt LowererImpl::lowerForall(Forall forall) {
  return Stmt();
}

Stmt LowererImpl::lowerForallDimension(Forall forall,
                                       std::vector<Iterator> locateIterators) {
  return Stmt();
}

Stmt LowererImpl::lowerForallCoordinate(Forall forall, Iterator iterator,
                                        std::vector<Iterator> locateIterators) {
  return Stmt();
}


Stmt LowererImpl::lowerForallPosition(Forall forall, Iterator iterator,
                                      std::vector<Iterator> locateIterators) {
  return Stmt();
}

Stmt LowererImpl::lowerForallMerge(Forall forall, MergeLattice lattice) {
  return Stmt();
}

Stmt LowererImpl::lowerWhere(Where where) {
  return Stmt();
}

Stmt LowererImpl::lowerMulti(Multi multi) {
  return Stmt();
}

Stmt LowererImpl::lowerSequence(Sequence sequence) {
  return Stmt();
}

Expr LowererImpl::lowerAccess(Access) {
  return Expr();
}

Expr LowererImpl::lowerLiteral(Literal) {
  return Expr();
}

Expr LowererImpl::lowerNeg(Neg) {
  return Expr();
}

Expr LowererImpl::lowerAdd(Add) {
  return Expr();
}

Expr LowererImpl::lowerSub(Sub) {
  return Expr();
}

Expr LowererImpl::lowerMul(Mul) {
  return Expr();
}

Expr LowererImpl::lowerDiv(Div) {
  return Expr();
}

Expr LowererImpl::lowerSqrt(Sqrt) {
  return Expr();
}

bool LowererImpl::generateAssembleCode() const {
  return this->assemble;
}

bool LowererImpl::generateComputeCode() const {
  return this->compute;
}

ir::Expr LowererImpl::getTensorVar(TensorVar tensorVar) const {
  taco_iassert(util::contains(this->tensorVars, tensorVar));
  return this->tensorVars.at(tensorVar);
}

ir::Expr LowererImpl::getDimension(IndexVar indexVar) const {
  taco_iassert(util::contains(this->dimensions, indexVar));
  return this->dimensions.at(indexVar);
}

Iterator LowererImpl::getIterator(ModeAccess modeAccess) const {
  taco_iassert(util::contains(this->iterators, modeAccess));
  return this->iterators.at(modeAccess);
}

ir::Expr LowererImpl::getCoordinateVar(IndexVar indexVar) const {
  taco_iassert(util::contains(this->coordVars, indexVar));
  return this->coordVars.at(indexVar);
}

ir::Expr LowererImpl::getCoordinateVar(Iterator iterator) const {
  taco_iassert(util::contains(this->indexVars, iterator) &&
               util::contains(this->coordVars, this->indexVars.at(iterator)));
  return this->coordVars.at(this->indexVars.at(iterator));
}

}
