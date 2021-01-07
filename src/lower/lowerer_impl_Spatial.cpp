#include <taco/lower/mode_format_compressed.h>
#include "taco/lower/lowerer_impl_Spatial.h"
#include "taco/lower/lowerer_impl.h"

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
           : GetProperty::make(getTensorVar(var), TensorProperty::Values, 0, var.getOrder());
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
} // namespace taco
